# 🤖 Chatbot AB Testing v2.0

Chatbot inteligente para análisis de experimentos A/B usando **Superlinked** para indexación vectorial multi-atributo y **OpenAI GPT-4** para procesamiento de lenguaje natural.

## 🚀 Características

- **🔍 Búsqueda semántica avanzada** con Superlinked
- **📊 Análisis estadístico completo** (lifts, significancia, segmentación)
- **🧠 Procesamiento de lenguaje natural** con OpenAI
- **⚙️ Configuración flexible** con Pydantic
- **🏗️ Arquitectura modular** y escalable

## 📁 Estructura del Proyecto

```
├── chatbot_v2.py          # Chatbot principal
├── config.py              # Configuración con Pydantic
├── schema.py              # Schema de Superlinked
├── index.py               # Índice vectorial multi-atributo
├── query.py               # Motor de consultas
├── data_processing.py     # Procesamiento de datos
├── analytics.py           # Análisis estadístico
├── requirements.txt       # Dependencias
├── tiendas_detalle.csv    # Datos del experimento
└── .env                   # Variables de entorno
```

## 🛠️ Instalación

1. **Instalar dependencias:**
```bash
pip install -r requirements.txt
```

2. **Configurar variables de entorno:**
Asegúrate de que tu archivo `.env` contenga:
```
OPENAI_API_KEY=tu_api_key_aqui
```

3. **Ejecutar el chatbot:**
```bash
python chatbot_v2.py
```

## 💻 Uso

### Ejemplos de Consultas

- **Análisis general:**
  - "¿Cuál fue el lift en conversiones del experimento?"
  - "¿Hay significancia estadística en los resultados?"

- **Análisis por segmentos:**
  - "¿Cómo se comportaron las tiendas Mall vs Street?"
  - "¿Qué región tuvo mejor performance?"
  - "¿Hay diferencias en el Norte vs Sur?"

- **Métricas específicas:**
  - "¿Cuál fue el revenue total del grupo control?"
  - "¿Qué variante tuvo mejor conversion rate?"

### Capacidades del Sistema

1. **Detección automática de filtros** en consultas naturales
2. **Búsqueda vectorial multi-atributo** usando Superlinked
3. **Análisis estadístico robusto** con scipy
4. **Segmentación automática** por región y tipo de tienda
5. **Respuestas contextualizadas** con OpenAI

## ⚙️ Configuración Avanzada

### Pesos de Espacios Vectoriales

Puedes ajustar los pesos en `config.py`:

```python
# Pesos configurables para espacios vectoriales
title_space_weight: float = 1.0
description_space_weight: float = 1.0
region_space_weight: float = 0.8
tipo_tienda_space_weight: float = 0.8
experimento_space_weight: float = 0.9
# ... más pesos
```

### Modelos de OpenAI

Cambiar modelo en `config.py`:
```python
openai_model_id: str = "gpt-4"  # o "gpt-3.5-turbo"
```

## 🧠 Arquitectura Técnica

### Superlinked Multi-Attribute Indexing

- **Espacios categóricos:** experimento, región, tipo_tienda
- **Espacios numéricos:** usuarios, conversiones, revenue, conversion_rate  
- **Espacios de texto:** descripciones con embeddings
- **Indexación combinada** para búsquedas complejas

### Pipeline de Consultas

1. **Extracción de filtros** automática desde consulta natural
2. **Búsqueda semántica** usando similitud vectorial
3. **Filtrado por atributos** específicos
4. **Análisis estadístico** contextual
5. **Generación de respuesta** con OpenAI

## 📊 Datos Soportados

El sistema analiza datos de AB Testing con:

- `experimento`: Control vs Experimento_A
- `tienda_id`: Identificador de tienda
- `region`: Este, Norte, Sur, Oeste
- `tipo_tienda`: Mall, Street, Outlet
- `usuarios`: Número de usuarios
- `conversiones`: Número de conversiones
- `revenue`: Ingresos generados
- `conversion_rate`: Tasa de conversión

## 🔧 Desarrollo

### Extender Funcionalidad

1. **Nuevos espacios vectoriales** en `index.py`
2. **Métricas adicionales** en `analytics.py`
3. **Tipos de consulta** en `query.py`
4. **Configuraciones** en `config.py`

### Testing

```bash
# Ejecutar tests (implementar según necesidad)
python -m pytest tests/
```

## 📈 Métricas y Análisis

El sistema calcula automáticamente:

- **Lifts porcentuales** en conversión y revenue
- **Significancia estadística** (t-test, chi-square)
- **Análisis por segmentos** (región, tipo de tienda)
- **Métricas agregadas** y promedios
- **Intervalos de confianza** y p-values

## 🤝 Contribuciones

1. Fork el repositorio
2. Crea una rama feature
3. Commit tus cambios
4. Push a la rama
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo licencia MIT. Ver `LICENSE` para más detalles.