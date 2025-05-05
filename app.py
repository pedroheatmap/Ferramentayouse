import pandas as pd
import numpy as np
from flask import Flask, render_template_string, request, send_file, jsonify
from io import BytesIO
from datetime import datetime
from geopy.geocoders import Nominatim
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import KDTree
import time
import warnings
import os
import logging
from geopy.distance import geodesic
import sys
import io
import json
from pathlib import Path
import gzip
from functools import lru_cache
import diskcache as dc
import hashlib
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash
from diskcache import FanoutCache
from functools import lru_cache
from diskcache import Cache

auth = HTTPBasicAuth()

users = {
    "Car10": generate_password_hash("Car10@2025")  # Substitua por uma senha forte
}

@auth.verify_password
def verify_password(username, password):
    if username in users and check_password_hash(users.get(username), password):
        return username

# Configuração otimizada para o Render
CACHE_DIR = Path('cache_data')
DISK_CACHE = FanoutCache(
    str(CACHE_DIR),
    shards=4,  # Melhor para concorrência
    timeout=1,
    size_limit=3e10,  # ~30GB
    disk_min_file_size=2**20  # 1MB
)

# Cache em memória para dados pequenos e frequentes
MEMORY_CACHE = {}

# Decorator combinado
def layered_cache(ttl=3600, maxsize=1024):
    def decorator(func):
        # Cache em memória (LRU)
        mem_cached = lru_cache(maxsize=maxsize)(func)
        
        # Cache em disco
        def wrapper(*args, **kwargs):
            cache_key = f"{func.__module__}_{func.__name__}_{get_cache_key(*args, **kwargs)}"
            
            # Tenta memória primeiro
            if cache_key in MEMORY_CACHE:
                return MEMORY_CACHE[cache_key]
                
            # Tenta disco
            result = DISK_CACHE.get(cache_key)
            if result is not None:
                MEMORY_CACHE[cache_key] = result  # Popula cache memória
                return result
                
            # Executa função
            result = mem_cached(*args, **kwargs)
            DISK_CACHE.set(cache_key, result, expire=ttl)
            MEMORY_CACHE[cache_key] = result
            return result
            
        wrapper.__name__ = func.__name__
        return wrapper
    return decorator

def get_cache_key(*args, **kwargs):
    """Gera uma chave única para os parâmetros"""
    key = hashlib.md5()
    for arg in args:
        key.update(str(arg).encode())
    for k, v in sorted(kwargs.items()):
        key.update(f"{k}={v}".encode())
    return key.hexdigest()

def cache_to_disk(ttl=432000):
    """Decorator para cache em disco com tempo de vida"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            cache_key = f"{func.__module__}_{func.__name__}_{get_cache_key(*args, **kwargs)}"
            
            # Tenta obter do cache
            result = DISK_CACHE.get(cache_key)
            if result is not None:
                return result
                
            # Executa a função se não estiver em cache
            result = func(*args, **kwargs)
            DISK_CACHE.set(cache_key, result, expire=ttl)
            return result
        
        # Preserva o nome original da função para o Flask
        wrapper.__name__ = func.__name__
        return wrapper
    return decorator

# Configuração para lidar com Unicode no Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('cobertura_oficinas.log', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
else:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('cobertura_oficinas.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.6f}'.format)

app = Flask(__name__, static_folder='static')

# ========== CONFIGURAÇÕES OTIMIZADAS ==========
RAIOS_KM = list(range(5, 101, 5))
CACHE_FILE = Path('geocode_cache.json')
LOTE_CLIENTES = 10000  # Reduzido para otimizar memória
MAX_CLIENTES_MAP = 3000  # Reduzido para o heatmap
GEOPY_TIMEOUT = 10
GEOPY_DELAY = 1
MEMORY_OPTIMIZED = True  # Ativar otimizações de memória

# ========== FUNÇÕES AUXILIARES ==========
def load_data_csv(filename, dtype=None):
    """Carrega dados de CSV com tratamento robusto de tipos e encoding"""
    try:
        # Lista de codificações a tentar (em ordem de preferência)
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
        
        # Verifica se existe versão compactada (prioridade)
        gz_file = f"{filename}.gz"
        if os.path.exists(gz_file):
            for encoding in encodings:
                try:
                    with gzip.open(gz_file, 'rt', encoding=encoding) as f:
                        df = pd.read_csv(f, sep=';', decimal=',')
                        # Conversão segura para float32
                        for col in ['Latitude', 'Longitude']:
                            if col in df.columns:
                                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), 
                                                     errors='coerce').astype('float32')
                        return df
                except (UnicodeDecodeError, pd.errors.ParserError):
                    continue
            raise UnicodeDecodeError("Nenhuma codificação funcionou para o arquivo compactado")
        elif os.path.exists(filename):
            for encoding in encodings:
                try:
                    df = pd.read_csv(filename, sep=';', decimal=',', encoding=encoding)
                    # Conversão segura para float32
                    for col in ['Latitude', 'Longitude']:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), 
                                                 errors='coerce').astype('float32')
                    return df
                except (UnicodeDecodeError, pd.errors.ParserError):
                    continue
            raise UnicodeDecodeError("Nenhuma codificação funcionou para o arquivo")
        else:
            raise FileNotFoundError(f"Arquivo {filename} não encontrado")
    except Exception as e:
        logging.error(f"Erro ao carregar {filename}: {str(e)}")
        raise
    
def get_clientes():
    if not hasattr(app, 'clientes_df'):
        # Usar tipos otimizados para memória
        dtype = {
            'Latitude': 'float32',
            'Longitude': 'float32'
        }
        app.clientes_df = load_data_csv("clientes.csv", dtype=dtype)
        app.clientes_df = converter_coordenadas(app.clientes_df)
        
        if MEMORY_OPTIMIZED:
            # Manter apenas colunas essenciais
            keep_cols = ['Latitude', 'Longitude']
            app.clientes_df = app.clientes_df[keep_cols].copy()
            
    return app.clientes_df

def load_cache():
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"Erro ao carregar cache: {str(e)} - Criando novo cache")
            return {}
    return {}

def save_cache(cache):
    try:
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.warning(f"Erro ao salvar cache: {str(e)}")

CACHE_CIDADES = load_cache()

def coordenada_no_brasil(lat, lon):
    """Verifica se as coordenadas estão dentro dos limites aproximados do Brasil"""
    return (lat >= -33.75 and lat <= 5.27 and 
            lon >= -73.99 and lon <= -34.79)

def converter_coordenadas(df):
    df = df.copy()

    for col in ['Latitude', 'Longitude']:
        if col not in df.columns:
            raise ValueError(f"Coluna obrigatória '{col}' não encontrada")

        # Garante que é string, substitui vírgula por ponto e converte para numérico
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(',', '.').str.strip(),
            errors='coerce'
        )

    # Remove linhas com valores nulos
    df = df.dropna(subset=['Latitude', 'Longitude'])
    
    # Filtra coordenadas válidas e dentro do Brasil
    valid_coords = df[
        (df['Latitude'].between(-90, 90)) &
        (df['Longitude'].between(-180, 180)) &
        df.apply(lambda x: coordenada_no_brasil(x['Latitude'], x['Longitude']), axis=1)
    ]

    if len(valid_coords) == 0:
        raise ValueError("Nenhuma coordenada válida encontrada após filtragem")

    if len(df) != len(valid_coords):
        logging.warning(f"Filtradas {len(df) - len(valid_coords)} coordenadas inválidas")

    return valid_coords

def get_cidade(lat, lon):
    chave = f"{lat:.4f},{lon:.4f}"
    if chave not in CACHE_CIDADES:
        try:
            geolocator = Nominatim(
                user_agent="cobertura_oficinas_app",
                timeout=GEOPY_TIMEOUT,
                domain="nominatim.openstreetmap.org"
            )
            location = geolocator.reverse((lat, lon), exactly_one=True, language='pt')
            
            if location:
                address = location.raw.get('address', {})
                cidade = (address.get('city') or 
                         address.get('town') or 
                         address.get('village') or 
                         address.get('municipality') or 
                         "Não identificado")
            else:
                cidade = "Não identificado"
                
            CACHE_CIDADES[chave] = cidade
            time.sleep(GEOPY_DELAY)
        except Exception as e:
            logging.warning(f"Erro ao obter cidade para {lat},{lon}: {str(e)}")
            CACHE_CIDADES[chave] = "Erro na consulta"
        save_cache(CACHE_CIDADES)
    return CACHE_CIDADES[chave]

@cache_to_disk(ttl=432000)
def calcular_distancia_lote(lote_clientes, oficinas):
    # Verificar tamanhos dos arrays primeiro
    if len(lote_clientes) == 0 or len(oficinas) == 0:
        return np.array([])
    
    # Converter para radianos em chunks
    lote_clientes = np.radians(np.asarray(lote_clientes, dtype=np.float32))
    oficinas = np.radians(np.asarray(oficinas, dtype=np.float32))
    
    # Dividir cálculo em partes menores
    try:
        dlat = oficinas[:, 0][:, np.newaxis] - lote_clientes[:, 0]
        dlon = oficinas[:, 1][:, np.newaxis] - lote_clientes[:, 1]
        
        a = np.sin(dlat/2)**2 + np.cos(lote_clientes[:, 0]) * np.cos(oficinas[:, 0][:, np.newaxis]) * np.sin(dlon/2)**2
        return 6371 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    except MemoryError:
        # Se falhar, calcular em pedaços menores
        distancias = []
        for i in range(0, len(oficinas), 100):  # Processar 100 oficinas por vez
            ofc = oficinas[i:i+100]
            dlat = ofc[:, 0][:, np.newaxis] - lote_clientes[:, 0]
            dlon = ofc[:, 1][:, np.newaxis] - lote_clientes[:, 1]
            a_part = np.sin(dlat/2)**2 + np.cos(lote_clientes[:, 0]) * np.cos(ofc[:, 0][:, np.newaxis]) * np.sin(dlon/2)**2
            distancias.append(6371 * 2 * np.arctan2(np.sqrt(a_part), np.sqrt(1 - a_part)))
        return np.hstack(distancias)


def calcular_cobertura_otima(clientes_coords, raio_km):
    if len(clientes_coords) == 0:
        return np.array([]), 0
    
    n_clusters = max(1, min(50, int(len(clientes_coords) / 500))) # Limite de 50 clusters
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    kmeans.fit(clientes_coords)
    centros = kmeans.cluster_centers_
    
    distancias = calcular_distancia_lote(clientes_coords, centros)
    centros_validos = []
    
    for i, centro in enumerate(centros):
        if not coordenada_no_brasil(centro[0], centro[1]):
            continue
        clientes_no_raio = np.sum(distancias[i] <= raio_km * 1.2)  # Buffer de 20%
        if clientes_no_raio > 50:  # Mínimo de 50 clientes
            centros_validos.append(centro)
    
    return np.array(centros_validos), len(centros_validos)

# ========== CARREGAMENTO DE DADOS ==========

print("⏳ Iniciando carregamento de dados...")

try:
    # Carregar dados com tratamento robusto
    oficinas_df = converter_coordenadas(load_data_csv("oficinas.csv"))
    clientes_df = converter_coordenadas(load_data_csv("clientes.csv"))
    
    if len(oficinas_df) == 0 or len(clientes_df) == 0:
        raise ValueError("Dados insuficientes após filtragem")
    
    # Converter para arrays numpy com tipos otimizados
    coords_oficinas = oficinas_df[['Latitude', 'Longitude']].values.astype(np.float32)
    coords_clientes = clientes_df[['Latitude', 'Longitude']].values.astype(np.float32)
    
    print(f"✅ Dados carregados: {len(oficinas_df)} oficinas e {len(clientes_df):,} clientes válidos")
    
except Exception as e:
    print(f"❌ Erro crítico ao carregar dados: {str(e)}")
    logging.error(f"❌ Erro crítico ao carregar dados: {str(e)}", exc_info=True)
    exit()


# ========== INTERFACE WEB ==========
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Análise de Cobertura - Oficinas</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css"/>
    <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
    <script src="https://unpkg.com/leaflet.heat@0.2.0/dist/leaflet-heat.js"></script>
    <style>
        :root {
            --primary-color: #e74c3c;
            --secondary-color: #c0392b;
            --danger-color: #e74c3c;
            --success-color: #27ae60;
            --warning-color: #f39c12;
            --text-color: #2c3e50;
            --bg-color: #f8f9fa;
        }
        body { 
            margin: 0; 
            padding: 0; 
            font-family: 'Arial', sans-serif; 
            background-color: #f5f5f5;
        }
        #container { 
            display: flex; 
            height: 100vh; 
        }
        #map { 
            flex: 3; 
        }
        #controls { 
            flex: 1; 
            padding: 20px; 
            background: white; 
            overflow-y: auto; 
            box-shadow: -2px 0 10px rgba(0,0,0,0.1);
        }
        .header {
            display: flex;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 2px solid var(--primary-color);
            width: 100%;
        }
        .logo {
            max-height: 60px;
            margin-right: 15px;
        }
        .title {
            color: var(--primary-color);
            margin: 0;
            font-size: 20px;
            font-weight: bold;
            text-transform: uppercase;
        }
        .slider-container { 
            margin-bottom: 20px; 
        }
        .slider { 
            width: 100%; 
            -webkit-appearance: none;
            height: 8px;
            border-radius: 4px;
            background: #ddd;
            outline: none;
        }
        .slider::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: var(--primary-color);
            cursor: pointer;
        }
        .value-display { 
            font-weight: bold; 
            color: var(--primary-color); 
            margin: 5px 0 15px;
            font-size: 18px;
        }
        .metric { 
            background: white; 
            padding: 15px; 
            border-radius: 8px; 
            margin-bottom: 20px; 
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            border-left: 4px solid var(--primary-color);
        }
        .metric h3 {
            color: var(--primary-color);
            margin-top: 0;
            border-bottom: 1px solid #eee;
            padding-bottom: 8px;
        }
        #export-btn, #sugerir-btn, #heatmap-btn, #raio-btn, #exportar-sugestoes-btn {
            color: white; 
            border: none;
            padding: 12px; 
            width: 100%; 
            border-radius: 6px; 
            cursor: pointer;
            font-weight: bold; 
            margin: 10px 0;
            font-size: 16px;
            transition: all 0.3s ease;
        }
        #export-btn {
            background: var(--primary-color);
        }
        #export-btn:hover { 
            background: var(--secondary-color);
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        #sugerir-btn { 
            background: var(--warning-color);
        }
        #sugerir-btn:hover { 
            background: #e67e22;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        #heatmap-btn {
            background: #3498db;
        }
        #heatmap-btn:hover {
            background: #2980b9;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        #raio-btn {
            background: #9b59b6;
        }
        #raio-btn:hover {
            background: #8e44ad;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        #exportar-sugestoes-btn {
            background-color: #f39c12;
        }
        #exportar-sugestoes-btn:hover {
            background-color: #e67e22;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .legend {
            padding: 10px;
            background: white;
            border-radius: 5px;
            box-shadow: 0 1px 5px rgba(0,0,0,0.2);
            font-size: 14px;
        }
        .loading {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.7);
            z-index: 1000;
            justify-content: center;
            align-items: center;
            color: white;
            font-size: 24px;
            flex-direction: column;
        }
        .loading-spinner {
            border: 5px solid #f3f3f3;
            border-top: 5px solid var(--primary-color);
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin-bottom: 15px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .sugestao-info {
            background-color: #fff3cd;
            padding: 12px;
            border-radius: 6px;
            margin-top: 10px;
            border-left: 4px solid var(--warning-color);
            font-size: 14px;
        }
        .popup-content {
            font-size: 14px;
            line-height: 1.5;
        }
        .popup-content b {
            color: var(--primary-color);
        }
        .spacer {
            flex-grow: 1;
        }
    </style>
</head>
<body>
    <div class="loading" id="loading">
        <div class="loading-spinner"></div>
        <div>Processando dados... Por favor aguarde</div>
    </div>
    
    <div id="container">
        <div id="map"></div>
        <div id="controls">
            <div class="header">
                <img src="{{ url_for('static', filename='logo.png') }}" alt="Logo" class="logo">
                <h1 class="title">HEATMAP ANALYZER</h1>
                <div class="spacer"></div>
                <img src="{{ url_for('static', filename='logo2.png') }}" alt="Logo 2" class="logo">
            </div>
            
            <div class="slider-container">
                <label>Raio de Atendimento (km):</label>
                <input type="range" min="5" max="100" step="5" value="15" id="raioSlider" class="slider">
                <div class="value-display" id="raio-value">15 km</div>
            </div>
            
            <div class="metric">
                <h3>📊 Métricas Atuais</h3>
                <p>👥 <strong>Clientes cobertos:</strong> <span id="cobertos">0</span></p>
                <p>🚫 <strong>Clientes descobertos:</strong> <span id="descobertos" style="color: var(--danger-color);">0</span></p>
                <p>📈 <strong>Percentual coberto:</strong> <span id="percentual">0%</span></p>
                <p>🏭 <strong>Oficinas ativas:</strong> <span id="oficinas-ativas">0</span></p>
            </div>
            
            <button id="sugerir-btn" onclick="sugerirOficinas()">
                🔍 Sugerir Novas Oficinas
            </button>
            
            <div id="sugestao-container" style="display: none;">
                <div class="metric">
                <h3>💡 Sugestão de Cobertura</h3>
               <div class="slider-container">
               <label>Número de Oficinas Sugeridas:</label>
               <input type="range" min="10" max="300" step="5" value="10" id="numOficinasSlider" class="slider">
               <div class="value-display" id="num-oficinas-value">10</div>
    </div>
               <p>📌 <strong>Oficinas Processadas:</strong> <span id="oficinas-sugeridas">0</span></p>
               <p>👥 <strong>Total de Beneficiados:</strong> <span id="clientes-adicionais">0</span></p>
               <p>👤 <strong>Clientes Novos cobertos:</strong> <span id="clientes-unicos">0</span></p>
             <div class="sugestao-info">
                 <p>Clique nos marcadores amarelos no mapa para ver as localizações sugeridas</p>
    </div>
</div>
            </div>
            
            <button id="heatmap-btn" onclick="toggleHeatmap()">
                🔥 Alternar Heatmap
            </button>
            
            <button id="raio-btn" onclick="toggleCirculosRaio()">
                🎯 Mostrar Raios de Atendimento
            </button>
            
            <button id="export-btn" onclick="exportarDados()">
                💾 Exportar Relatório Completo
            </button>

            <button id="exportar-sugestoes-btn" onclick="exportarSugestoes()">
                📤 Exportar Sugestões de Oficinas
            </button>
        </div>
    </div>

    <script>
        // Dados para o frontend
        const oficinas = {{ oficinas|tojson }};
        const totalClientes = {{ total_clientes }};
        let currentRaio = 15;
        let sugestoes = [];
        let heatmapLayer = null;
        let circulosRaio = L.layerGroup();
        
        // Mapa
        const map = L.map('map').setView([-15.5, -55], 4.5);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '&copy; OpenStreetMap'
        }).addTo(map);
        
        // Legenda
        const legend = L.control({position: 'bottomright'});
        legend.onAdd = function(map) {
            const div = L.DomUtil.create('div', 'legend');
            div.innerHTML = `
                <div><i style="background:#27ae60; width:12px; height:12px; display:inline-block; border-radius:50%;"></i> Oficina</div>
                <div><i style="background:#f39c12; width:12px; height:12px; display:inline-block; border-radius:50%;"></i> Sugestão</div>
                <div><i style="background:#9b59b6; width:12px; height:12px; display:inline-block; border-radius:50%;"></i> Raio de Atendimento</div>
            `;
            return div;
        };
        legend.addTo(map);
        
        // Camadas
        const marcadores = L.layerGroup().addTo(map);
        const marcadores_sugestao = L.layerGroup().addTo(map);
        
        // Mostrar loading
        function showLoading() {
            document.getElementById('loading').style.display = 'flex';
        }
        
        // Esconder loading
        function hideLoading() {
            document.getElementById('loading').style.display = 'none';
        }
        
        // Exportar sugestões
        function exportarSugestoes() {
            showLoading();
            window.open(`/exportar_sugestoes?raio=${currentRaio}`, '_blank');
            setTimeout(hideLoading, 2000);
        }

        // Atualizar visualização
        function atualizarVisualizacao(raio) {
            showLoading();
            currentRaio = raio;
            document.getElementById('sugestao-container').style.display = 'none';
            marcadores_sugestao.clearLayers();
            marcadores.clearLayers();
            circulosRaio.clearLayers();
            
            document.getElementById('raio-value').textContent = `${raio} km`;
            
            fetch(`/cobertura?raio=${raio}`)
                .then(response => response.json())
                .then(data => {
                    // Adicionar marcadores (todas ativas)
                    data.oficinas_ativas.forEach(oficina => {
                        L.circleMarker(
                            [oficina.lat, oficina.lon], 
                            {
                                radius: 6 + Math.min(Math.log(oficina.clientes)/2, 5),
                                fillColor: '#27ae60',
                                color: '#fff',
                                weight: 1,
                                fillOpacity: 0.8
                            }
                        ).bindPopup(`
                            <div class="popup-content">
                                <b>${oficina.nome || 'Oficina ' + oficina.id}</b><br>
                                📍 ${oficina.lat.toFixed(4)}, ${oficina.lon.toFixed(4)}<br>
                                🔴 <b>Raio:</b> ${raio} km<br>
                                👥 <b>Clientes:</b> ${oficina.clientes.toLocaleString()}
                            </div>
                        `).addTo(marcadores);
                    });
                    
                    // Atualizar métricas
                    document.getElementById('cobertos').textContent = data.clientes_cobertos.toLocaleString();
                    document.getElementById('descobertos').textContent = (totalClientes - data.clientes_cobertos).toLocaleString();
                    document.getElementById('percentual').textContent = `${((data.clientes_cobertos / totalClientes) * 100).toFixed(1)}%`;
                    document.getElementById('oficinas-ativas').textContent = data.oficinas_ativas.length;
                    
                    // Ajustar zoom
                    if (data.oficinas_ativas.length > 0) {
                        const grupo = new L.featureGroup(marcadores.getLayers());
                        map.fitBounds(grupo.getBounds().pad(0.2));
                    }
                    
                    hideLoading();
                })
                .catch(error => {
                    console.error('Erro:', error);
                    hideLoading();
                    alert('Erro ao carregar dados. Por favor tente novamente.');
                });
        }
        
        // Filtrar sugestões com base no número selecionado
      // Na função filtrarSugestoes(), atualize para usar os novos dados:
      
function filtrarSugestoes(numOficinas) {
    if (sugestoes.length === 0) return;
    
    const sugestoesFiltradas = [...sugestoes].sort((a, b) => b.score - a.score).slice(0, numOficinas);
    
    marcadores_sugestao.clearLayers();
    
    showLoading();
    
    // Calcular totais
    let totalBeneficiados = 0;
    let clientesUnicos = 0;
    
    sugestoesFiltradas.forEach(sugestao => {
        totalBeneficiados += sugestao.clientes_potenciais;
        clientesUnicos += sugestao.clientes_unicos;
    });
    
    // Atualizar marcadores
    sugestoesFiltradas.forEach((sugestao, idx) => {
        L.circleMarker(
            [sugestao.lat, sugestao.lon], 
            {
                radius: 8,
                fillColor: '#f39c12',
                color: '#fff',
                weight: 2,
                fillOpacity: 0.9
            }
        ).bindPopup(`
            <div class="popup-content">
                <b>📍 Sugestão ${idx + 1}</b><br>
                Latitude: ${sugestao.lat.toFixed(4)}<br>
                Longitude: ${sugestao.lon.toFixed(4)}<br>
                🔴 <b>Raio de cobertura:</b> ${currentRaio} km<br>
                👥 <b>Clientes no raio (total):</b> ${sugestao.clientes_potenciais.toLocaleString()}<br>
                👤 <b>Clientes novos cobertos:</b> ${sugestao.clientes_unicos.toLocaleString()}<br>
                 ${sugestao.cidade ? `🏙️ <b>Cidade estimada:</b> ${sugestao.cidade}<br>` : ''}
            </div>
        `).addTo(marcadores_sugestao);
    });
    
    // Atualizar métricas
    document.getElementById('oficinas-sugeridas').textContent = sugestoesFiltradas.length;
    document.getElementById('clientes-adicionais').textContent = totalBeneficiados.toLocaleString();
    document.getElementById('clientes-unicos').textContent = clientesUnicos.toLocaleString();
    
    if (sugestoesFiltradas.length > 0) {
        const grupo = new L.featureGroup(marcadores_sugestao.getLayers());
        map.fitBounds(grupo.getBounds().pad(0.5));
    }
    
    hideLoading();
}

// Atualizar a função sugerirOficinas() para lidar com a nova resposta:
function sugerirOficinas() {
    showLoading();
    fetch(`/sugerir_oficinas?raio=${currentRaio}`)
        .then(response => response.json())
        .then(data => {
            sugestoes = data.sugestoes;
            // Adicionar os totais ao objeto sugestoes para acesso fácil
            sugestoes.total_beneficiados = data.total_beneficiados;
            sugestoes.clientes_unicos_totais = data.clientes_unicos_totais;
            
            const numOficinas = parseInt(document.getElementById('numOficinasSlider').value);
            filtrarSugestoes(numOficinas);
            
            document.getElementById('sugestao-container').style.display = 'block';
            hideLoading();
        })
        .catch(error => {
            console.error('Erro:', error);
            hideLoading();
            alert('Erro ao calcular sugestões. Por favor tente novamente.');
        });
}
        
        // Função para calcular clientes únicos cobertos
        function calcularClientesUnicos(sugestoesFiltradas) {
            showLoading();
            return fetch(`/calcular_clientes_unicos?raio=${currentRaio}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    sugestoes: sugestoesFiltradas
                })
            })
            .then(response => response.json())
            .then(data => {
                hideLoading();
                return data.clientes_unicos;
            })
            .catch(error => {
                console.error('Erro ao calcular clientes únicos:', error);
                hideLoading();
                return 0;
            });
        }
        
        // Sugerir novas oficinas
        function sugerirOficinas() {
            showLoading();
            fetch(`/sugerir_oficinas?raio=${currentRaio}`)
                .then(response => response.json())
                .then(data => {
                    sugestoes = data.sugestoes;
                    const numOficinas = parseInt(document.getElementById('numOficinasSlider').value);
                    filtrarSugestoes(numOficinas);
                    
                    document.getElementById('sugestao-container').style.display = 'block';
                    hideLoading();
                })
                .catch(error => {
                    console.error('Erro:', error);
                    hideLoading();
                    alert('Erro ao calcular sugestões. Por favor tente novamente.');
                });
        }
        
        //Clientes Beneficiados
        function calcularClientesBeneficiados(sugestoesFiltradas) {
        showLoading();
         return fetch(`/calcular_clientes_beneficiados?raio=${currentRaio}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            sugestoes: sugestoesFiltradas
        })
    })
         .then(response => response.json())
         .then(data => {
         hideLoading();
         return {
            totalBeneficiados: data.total_beneficiados,
            clientesUnicos: data.clientes_unicos
        };
    })
          .catch(error => {
         console.error('Erro ao calcular clientes:', error);
         hideLoading();
         return {
            totalBeneficiados: 0,
            clientesUnicos: 0
        };
    });
}
        
        // Alternar heatmap
        function toggleHeatmap() {
            if (heatmapLayer) {
                map.removeLayer(heatmapLayer);
                heatmapLayer = null;
            } else {
                showLoading();
                
                fetch('/clientes_heatmap')
                    .then(response => response.json())
                    .then(data => {
                        heatmapLayer = L.heatLayer(data.clientes, {
                            radius: 25,
                            blur: 20,
                            maxZoom: 15,
                            minOpacity: 0.5,
                            gradient: {
                                0.1: 'blue',
                                0.3: 'cyan',
                                0.5: 'lime',
                                0.7: 'yellow',
                                1.0: 'red'
                            }
                        }).addTo(map);
                        
                        hideLoading();
                    })
                    .catch(error => {
                        console.error('Erro ao carregar heatmap:', error);
                        hideLoading();
                        alert('Erro ao carregar dados para heatmap');
                    });
            }
        }
        
        // Alternar círculos de raio
        function toggleCirculosRaio() {
            if (map.hasLayer(circulosRaio)) {
                map.removeLayer(circulosRaio);
                document.getElementById('raio-btn').textContent = '🎯 Mostrar Raios de Atendimento';
            } else {
                showLoading();
                circulosRaio.clearLayers();
                
                // Adiciona círculos para cada oficina ativa
                marcadores.getLayers().forEach(marker => {
                    const latlng = marker.getLatLng();
                    L.circle(latlng, {
                        radius: currentRaio * 1000,
                        color: '#9b59b6',
                        fillColor: '#9b59b6',
                        fillOpacity: 0.1,
                        weight: 1
                    }).addTo(circulosRaio);
                });
                
                circulosRaio.addTo(map);
                document.getElementById('raio-btn').textContent = '❌ Ocultar Raios';
                hideLoading();
            }
        }
        
        // Exportar dados
        function exportarDados() {
            showLoading();
            window.open(`/exportar?raio=${currentRaio}`, '_blank');
            setTimeout(hideLoading, 2000);
        }
        
        // Event listeners
        document.getElementById('raioSlider').addEventListener('input', function() {
            atualizarVisualizacao(parseInt(this.value));
        });
        
        document.getElementById('numOficinasSlider').addEventListener('input', function() {
        const numOficinas = Math.round(parseInt(this.value)/5)*5; // Arredonda para o múltiplo de 5 mais próximo
        document.getElementById('num-oficinas-value').textContent = numOficinas;
        filtrarSugestoes(numOficinas);
});
        // Inicialização
        atualizarVisualizacao(15);
    </script>
</body>
</html>
"""

# ========== ROTAS ==========

@app.route('/')
@auth.login_required
def index():
    oficinas_json = [{
        'id': idx + 1,
        'Nome': row.get('Nome', f'Oficina {idx+1}'),
        'Latitude': float(row['Latitude']),
        'Longitude': float(row['Longitude'])
    } for idx, row in oficinas_df.iterrows()]
    
    return render_template_string(
        HTML_TEMPLATE,
        oficinas=oficinas_json,
        total_clientes=len(clientes_df)
    )

@app.route('/cobertura')
@cache_to_disk(ttl=432000)  # Cache por 10 minutos
def obter_cobertura():
    try:
        raio = int(request.args.get('raio', 15))
        if raio <= 0:
            raise ValueError("Raio deve ser positivo")
            
        # Adicionar timeout
        start_time = time.time()
        timeout = 29  # Segundos (menos que o timeout do Gunicorn)
        
        clientes_cobertos = 0
        clientes_por_oficina = np.zeros(len(oficinas_df), dtype=np.int32)
        
        for i in range(0, len(coords_clientes), LOTE_CLIENTES):
            if time.time() - start_time > timeout:
                raise TimeoutError("Processamento excedeu o tempo limite")
                
            lote = coords_clientes[i:i+LOTE_CLIENTES]
            distancias = calcular_distancia_lote(lote, coords_oficinas)
            
            if distancias.size == 0:
                continue
                
            clientes_cobertos += np.sum(np.any(distancias <= raio, axis=0))
            clientes_por_oficina += np.sum(distancias <= raio, axis=1)
            
            # Clientes por oficina (dentro do raio)
            clientes_por_oficina += np.sum(distancias <= raio, axis=1)
        
        # Preparar resposta
        ativas = []
        for idx, row in oficinas_df.iterrows():
            if idx < len(clientes_por_oficina) and clientes_por_oficina[idx] > 0:
                ativas.append({
                    'id': idx + 1,
                    'nome': row.get('Nome', f'Oficina {idx+1}'),
                    'lat': float(row['Latitude']),
                    'lon': float(row['Longitude']),
                    'clientes': int(clientes_por_oficina[idx])
                })
        
        return jsonify({
            'clientes_cobertos': int(clientes_cobertos),
            'oficinas_ativas': ativas,
            'oficinas_inativas': []
        })
    
    except Exception as e:
        logging.error(f"Erro em /cobertura: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500
    
@app.route('/calcular_clientes_beneficiados', methods=['POST'])
def calcular_clientes_beneficiados():
    try:
        raio = int(request.args.get('raio'))
        data = request.get_json()
        sugestoes = data.get('sugestoes', [])
        
        if not sugestoes:
            return jsonify({
                'total_beneficiados': 0,
                'clientes_unicos': 0
            })
        
        # Identificar clientes não cobertos pelas oficinas existentes
        mascara_nao_cobertos = np.ones(len(coords_clientes), dtype=bool)
        
        for i in range(0, len(coords_clientes), LOTE_CLIENTES):
            lote = coords_clientes[i:i+LOTE_CLIENTES]
            distancias = calcular_distancia_lote(lote, coords_oficinas)
            mascara_nao_cobertos[i:i+len(lote)] = ~np.any(distancias <= raio, axis=0)
        
        clientes_nao_cobertos = coords_clientes[mascara_nao_cobertos]
        
        if len(clientes_nao_cobertos) == 0:
            return jsonify({
                'total_beneficiados': 0,
                'clientes_unicos': 0
            })
        
        # Calcular total de clientes beneficiados (SOMANDO TODOS, MESMO COM SOBREPOSIÇÃO)
        total_beneficiados = 0
        clientes_cobertos_mask = np.zeros(len(clientes_nao_cobertos), dtype=bool)
        
        for sugestao in sugestoes:
            centro = np.array([[sugestao['lat'], sugestao['lon']]])
            distancias = calcular_distancia_lote(clientes_nao_cobertos, centro)
            no_raio = (distancias <= raio).flatten()
            total_beneficiados += np.sum(no_raio)  # Soma todos, mesmo os já contados
            clientes_cobertos_mask |= no_raio  # Para clientes únicos
        
        return jsonify({
            'total_beneficiados': int(total_beneficiados),  # Soma de todos os clientes em todos os raios
            'clientes_unicos': int(np.sum(clientes_cobertos_mask))  # Clientes únicos cobertos
        })
        
    except Exception as e:
        logging.error(f"Erro em /calcular_clientes_beneficiados: {str(e)}")
        return jsonify({
            'total_beneficiados': 0,
            'clientes_unicos': 0,
            'error': str(e)
        }), 500

@app.route('/sugerir_oficinas')
@cache_to_disk(ttl=432000)  # Cache por 30 minutos
def sugerir_oficinas():
    try:
        raio = int(request.args.get('raio'))
        
        # 1. Identificar clientes NÃO cobertos pelas oficinas existentes
        mascara_nao_cobertos = np.ones(len(coords_clientes), dtype=bool)
        
        for i in range(0, len(coords_clientes), LOTE_CLIENTES):
            lote = coords_clientes[i:i+LOTE_CLIENTES]
            distancias = calcular_distancia_lote(lote, coords_oficinas)
            mascara_nao_cobertos[i:i+len(lote)] = ~np.any(distancias <= raio, axis=0)
        
        clientes_nao_cobertos = coords_clientes[mascara_nao_cobertos]
        
        if len(clientes_nao_cobertos) == 0:
            return jsonify({
                'sugestoes': [], 
                'total_beneficiados': 0,
                'clientes_unicos_totais': 0
            })
        
        # 2. Usar DBSCAN para encontrar áreas densas de clientes descobertos
        eps = raio * 0.007
        dbscan = DBSCAN(eps=eps, min_samples=30)
        clusters = dbscan.fit_predict(clientes_nao_cobertos)
        
        # 3. Calcular centroides e número de clientes por cluster
        sugestoes = []
        todas_sugestoes_coords = []
        
        for cluster_id in np.unique(clusters):
            if cluster_id == -1:  # Ignorar outliers
                continue
                
            mascara_cluster = (clusters == cluster_id)
            clientes_cluster = clientes_nao_cobertos[mascara_cluster]
                        
            # Calcular centroide como a média das coordenadas 
            centro = np.mean(clientes_cluster, axis=0)
            lat, lon = centro[0], centro[1]
            
            if not coordenada_no_brasil(lat, lon):
                continue
                
            # Calcular TOTAL de clientes no raio (incluindo os já cobertos)
            distancias_total = calcular_distancia_lote(coords_clientes, [centro])
            total_beneficiados = np.sum(distancias_total <= raio)
            
            # Calcular CLIENTES NOVOS (apenas os que estavam descobertos)
            distancias_novos = calcular_distancia_lote(clientes_nao_cobertos, [centro])
            clientes_novos = np.sum(distancias_novos <= raio)
            
            sugestoes.append({
                'lat': float(lat),
                'lon': float(lon),
                'clientes_potenciais': int(total_beneficiados),  # Todos no raio
                'clientes_unicos': int(clientes_novos),  # Apenas os novos
                'score': float(clientes_novos)  # Prioriza áreas com mais clientes novos
            })
        
        # Ordenar por número de clientes novos e limitar a 100 melhores
        sugestoes = sorted(sugestoes, key=lambda x: -x['score'])[:300]
        
        # Calcular totais
        total_beneficiados = sum(s['clientes_potenciais'] for s in sugestoes)
        clientes_unicos_totais = sum(s['clientes_unicos'] for s in sugestoes)
        
        return jsonify({
            'sugestoes': sugestoes,
            'total_beneficiados': int(total_beneficiados),
            'clientes_unicos_totais': int(clientes_unicos_totais)
        })
    
    except Exception as e:
        logging.error(f"Erro em /sugerir_oficinas: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/clientes_heatmap')
@cache_to_disk(ttl=432000)  # Cache por 30 minutos
def clientes_heatmap():

    try:
        sample_size = min(10000, len(clientes_df))
        clientes_sample = clientes_df.sample(sample_size) if len(clientes_df) > sample_size else clientes_df
        
        # Calcula densidade para intensidade do heatmap
        coords = clientes_sample[['Latitude', 'Longitude']].values
        intensities = np.ones(len(coords)) * 0.8  # Valor base
        
        # Aumenta intensidade em áreas densas
        if len(coords) > 1000:
            tree = KDTree(coords)
            for i, point in enumerate(coords):
                intensities[i] += min(tree.query_radius([point], r=0.1, count_only=True)[0] / 10, 3)
        
        points = [[lat, lon, inten] for (lat, lon), inten in zip(coords, intensities)]
        
        return jsonify({'clientes': points})
    
    except Exception as e:
        logging.error(f"Erro em /clientes_heatmap: {str(e)}")
        return jsonify({'error': str(e)}), 500
    
@app.route('/exportar_sugestoes')
def exportar_sugestoes():
    try:
        raio = int(request.args.get('raio'))
        
        # Identificar clientes não cobertos
        mascara_nao_cobertos = np.ones(len(coords_clientes), dtype=bool)
        
        for i in range(0, len(coords_clientes), LOTE_CLIENTES):
            lote = coords_clientes[i:i+LOTE_CLIENTES]
            distancias = calcular_distancia_lote(lote, coords_oficinas)
            mascara_nao_cobertos[i:i+len(lote)] = ~np.any(distancias <= raio, axis=0)
        
        clientes_nao_cobertos = coords_clientes[mascara_nao_cobertos]
        
        # Reutilizar a lógica de sugestões
        response = sugerir_oficinas()
        if response.status_code != 200:
            return response
        
        sugestoes_data = response.get_json()
        sugestoes = sugestoes_data.get('sugestoes', [])
        
        # Calcular clientes únicos que seriam cobertos
        clientes_cobertos_mask = np.zeros(len(clientes_nao_cobertos), dtype=bool)
        for sugestao in sugestoes:
            centro = np.array([[sugestao['lat'], sugestao['lon']]])
            distancias = calcular_distancia_lote(clientes_nao_cobertos, centro)
            clientes_cobertos_mask |= (distancias <= raio).flatten()
        
        clientes_unicos_cobertos = np.sum(clientes_cobertos_mask)
        
        # Criar DataFrame
        df_sugestoes = pd.DataFrame(sugestoes)
        df_sugestoes['Cidade_Estimada'] = df_sugestoes.apply(
            lambda row: get_cidade(row['lat'], row['lon']), axis=1)
        
        # Adicionar sheet com métricas consistentes
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_sugestoes.to_excel(writer, sheet_name='Sugestoes_Oficinas', index=False)
            
            pd.DataFrame({
                'Métrica': [
                    'Raio (km)',
                    'Clientes Descobertos',
                    'Clientes que Serão Cobertos (únicos)',
                    'Sugestões Geradas',
                    'Data'
                ],
                'Valor': [
                    raio,
                    len(clientes_nao_cobertos),
                    clientes_unicos_cobertos,
                    len(sugestoes),
                    datetime.now().strftime('%d/%m/%Y %H:%M')
                ]
            }).to_excel(writer, sheet_name='Resumo', index=False)
        
        output.seek(0)
        return send_file(output, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                        as_attachment=True,
                        download_name=f"sugestoes_oficinas_{raio}km_{datetime.now().strftime('%Y%m%d')}.xlsx")
    except Exception as e:
        logging.error(f"Erro em /exportar_sugestoes: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500
    
    
@app.route('/calcular_clientes_unicos', methods=['POST'])
def calcular_clientes_unicos():
    try:
        raio = int(request.args.get('raio'))
        data = request.get_json()
        sugestoes = data.get('sugestoes', [])
        
        if not sugestoes:
            return jsonify({'clientes_unicos': 0})
        
        # Identificar clientes não cobertos pelas oficinas existentes
        mascara_nao_cobertos = np.ones(len(coords_clientes), dtype=bool)
        
        for i in range(0, len(coords_clientes), LOTE_CLIENTES):
            lote = coords_clientes[i:i+LOTE_CLIENTES]
            distancias = calcular_distancia_lote(lote, coords_oficinas)
            mascara_nao_cobertos[i:i+len(lote)] = ~np.any(distancias <= raio, axis=0)
        
        clientes_nao_cobertos = coords_clientes[mascara_nao_cobertos]
        
        if len(clientes_nao_cobertos) == 0:
            return jsonify({'clientes_unicos': 0})
        
        # Verificar quais clientes não cobertos seriam alcançados pelas sugestões
        clientes_cobertos_mask = np.zeros(len(clientes_nao_cobertos), dtype=bool)
        
        for sugestao in sugestoes:
            centro = np.array([[sugestao['lat'], sugestao['lon']]])
            distancias = calcular_distancia_lote(clientes_nao_cobertos, centro)
            clientes_cobertos_mask |= (distancias <= raio).flatten()
        
        return jsonify({
            'clientes_unicos': int(np.sum(clientes_cobertos_mask))
        })
        
    except Exception as e:
        logging.error(f"Erro em /calcular_clientes_unicos: {str(e)}")
        return jsonify({'error': str(e)}), 500
    
@app.route('/manutencao/limpar_cache')
@auth.login_required
def limpar_cache():
    try:
        DISK_CACHE.clear()
        MEMORY_CACHE.clear()
        return jsonify({'status': 'success', 'message': 'Cache limpo'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
    
@app.route('/manutencao/status_cache')
@auth.login_required
def status_cache():
    stats = {
        'disk_cache_size': DISK_CACHE.volume(),
        'disk_cache_items': DISK_CACHE.stats()['count'],
        'memory_cache_items': len(MEMORY_CACHE),
        'cache_dir': str(CACHE_DIR)
    }
    return jsonify(stats)
    

@app.route('/exportar')
def exportar():
    try:
        raio = int(request.args.get('raio'))
        clientes_cobertos = 0
        clientes_por_oficina = np.zeros(len(oficinas_df), dtype=np.int32)
        
        # Processar em lotes menores para otimizar memória
        LOTE = min(LOTE_CLIENTES, 10000)
        
        for i in range(0, len(coords_clientes), LOTE):
            lote_clientes = coords_clientes[i:i+LOTE]
            distancias = calcular_distancia_lote(lote_clientes, coords_oficinas)
            
            if distancias.size == 0:
                continue
                
            clientes_cobertos += np.sum(np.any(distancias <= raio, axis=1))
            clientes_por_oficina += np.sum(distancias <= raio, axis=0)
        
        # Preparar dados para exportação CSV
        df_oficinas = oficinas_df.copy()
        df_oficinas['Clientes_Cobertos'] = clientes_por_oficina
        
        # Gerar CSV em memória
        output = BytesIO()
        
        # Escrever métricas como primeiro CSV
        metricas = pd.DataFrame({
            'Métrica': ['Total Clientes', 'Clientes Cobertos', '% Cobertura', 
                       'Oficinas Ativas', 'Raio (km)', 'Data da Análise'],
            'Valor': [
                len(clientes_df),
                int(clientes_cobertos),
                f"{(clientes_cobertos/len(clientes_df))*100:.1f}%",
                len(oficinas_df),
                raio,
                datetime.now().strftime('%d/%m/%Y %H:%M')
            ]
        })
        
        # Compactar os dados para economizar memória
        with gzip.GzipFile(fileobj=output, mode='wb') as gz:
            metricas.to_csv(gz, index=False, encoding='utf-8')
            gz.write(b'\n\n')  # Separador
            df_oficinas.to_csv(gz, index=False, encoding='utf-8')
        
        output.seek(0)
        
        return send_file(
            output,
            mimetype='application/gzip',
            as_attachment=True,
            download_name=f"cobertura_oficinas_{raio}km_{datetime.now().strftime('%Y%m%d')}.csv.gz"
        )
    
    except Exception as e:
        logging.error(f"Erro em /exportar: {str(e)}", exc_info=True)
        return jsonify({'error': f"Erro ao exportar: {str(e)}"}), 500

if __name__ == '__main__':
    if not os.path.exists('static'):
        os.makedirs('static')
    
    if IS_RENDER:
        # Configurações otimizadas para produção no Render
        from waitress import serve
        logging.info("\n🚀 Aplicação pronta no Render! Acesse https://seu-app.onrender.com")
        serve(
            app,
            host='0.0.0.0',
            port=int(os.environ.get('PORT', 5000)),
            threads=8,  # Aumentado para o plano profissional
            channel_timeout=60,
            cleanup_interval=30
        )
    else:
        # Configuração local
        app.run(debug=False, threaded=True)