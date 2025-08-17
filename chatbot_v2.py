"""
Chatbot de AB Testing refactorizado con Superlinked
Versión modular siguiendo mejores prácticas
"""

from superlinked.framework.dsl.executor.in_memory.in_memory_executor import InMemoryExecutor
from openai import OpenAI
from typing import Dict, List, Any
import sys
import traceback

# Imports de módulos locales
from config import settings
from index import create_ab_testing_index
from query import ABTestingQueryEngine
from data_processing import ABTestingDataProcessor
from analytics import ABTestingAnalytics
from intent_router import IntentRouter

class ABTestingChatbotV2:
    """Chatbot de AB Testing refactorizado con arquitectura modular"""
    
    def __init__(self):
        print("🚀 Inicializando Chatbot de AB Testing v2.0...")
        
        # Configuración
        self.settings = settings
        self.openai_client = OpenAI(api_key=settings.openai_api_key.get_secret_value())
        
        # Procesamiento de datos
        print("📊 Cargando y procesando datos...")
        self.data_processor = ABTestingDataProcessor()
        self.analytics = ABTestingAnalytics(self.data_processor.data)
        
        # Crear índice Superlinked
        print("🔍 Creando índice vectorial Superlinked...")
        self.index, self.schema, self.spaces = create_ab_testing_index()
        
        # Configurar source y executor
        from superlinked.framework.dsl.source.in_memory_source import InMemorySource
        
        self.source = InMemorySource(self.schema)
        self.executor = InMemoryExecutor(sources=[self.source], indices=[self.index])
        self.app = self.executor.run()
        
        # Indexar documentos
        documents = self.data_processor.prepare_documents()
        if documents:
            self.source.put(documents)
        
        # Motor de consultas
        self.query_engine = ABTestingQueryEngine(self.index, self.schema, self.spaces, self.app)
        
        # Router de intents
        self.intent_router = IntentRouter(self.data_processor, self.settings)
        
        # Cache de análisis
        self._ab_analysis_cache = None
        
        print("✅ Chatbot inicializado correctamente!")
        self._print_data_summary()
    
    def _print_data_summary(self):
        """Imprime resumen de los datos cargados"""
        summary = self.data_processor.get_data_summary()
        print(f"""
📈 DATOS CARGADOS:
• Total registros: {summary['total_records']}
• Experimentos: {', '.join(summary['experiments'])}
• Regiones: {', '.join(summary['regions'])}
• Tipos de tienda: {', '.join(summary['store_types'])}
• Total usuarios: {summary['total_users']:,}
• Total conversiones: {summary['total_conversions']:,}
• Revenue total: ${summary['total_revenue']:,.2f}
""")
    
    def get_ab_analysis(self) -> Dict[str, Any]:
        """Obtiene análisis completo de AB Testing (con cache)"""
        if self._ab_analysis_cache is None:
            self._ab_analysis_cache = self.analytics.analyze_ab_test_results()
        return self._ab_analysis_cache
    
    def generate_response(self, user_query: str) -> str:
        """Genera respuesta usando OpenAI con contexto enriquecido"""
        try:
            # Usar el router de intents para consultas simples
            simple_response = self.intent_router.route_query(user_query)
            if simple_response:
                return simple_response
            
            # Extraer filtros automáticamente
            filters = self.query_engine.extract_filters_from_query(user_query)
            
            # Detectar si es una consulta de performance (mayor/menor/top/mejor)
            performance_results = []
            if self._is_performance_query(user_query):
                performance_results = self._handle_performance_query(user_query, filters)
            
            # Búsqueda semántica
            semantic_results = self.query_engine.semantic_search(
                user_query, 
                filters=filters,
                limit=self.settings.default_limit
            )
            
            # Búsqueda por filtros si los hay
            filter_results = []
            if filters:
                filter_results = self.query_engine.filter_search(**filters)
            
            # Obtener análisis completo
            ab_analysis = self.get_ab_analysis()
            
            # Crear contexto enriquecido
            context = self._build_context(user_query, semantic_results, filter_results, ab_analysis, performance_results)
            
            # Generar respuesta con OpenAI
            response = self.openai_client.chat.completions.create(
                model=self.settings.openai_model_id,
                messages=[
                    {
                        "role": "system", 
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user", 
                        "content": context
                    }
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            error_msg = f"Error generando respuesta: {str(e)}"
            print(f"❌ {error_msg}")
            traceback.print_exc()
            return f"Lo siento, ocurrió un error al procesar tu consulta: {error_msg}"
    
    def _is_performance_query(self, query: str) -> bool:
        """Detecta si es una consulta de performance/ranking"""
        query_lower = query.lower()
        performance_keywords = [
            "mayor", "menor", "máximo", "mínimo", "mejor", "peor",
            "top", "ranking", "más alto", "más bajo", "highest", "lowest",
            "best", "worst", "máx", "mín"
        ]
        return any(keyword in query_lower for keyword in performance_keywords)
    
    def _handle_performance_query(self, query: str, filters: Dict) -> List[Dict]:
        """Maneja consultas de performance específicas"""
        query_lower = query.lower()
        
        # Determinar métrica y orden
        metric = 'conversion_rate'  # Default
        order = 'desc'  # Default para "mayor/mejor"
        
        if any(word in query_lower for word in ["revenue", "ingreso", "ganancia"]):
            metric = 'revenue'
        elif any(word in query_lower for word in ["usuario", "users", "tráfico"]):
            metric = 'usuarios'
        elif any(word in query_lower for word in ["conversion", "conversión"]):
            metric = 'conversion_rate'
        
        if any(word in query_lower for word in ["menor", "mínimo", "peor", "lowest", "worst", "mín"]):
            order = 'asc'
        
        # Obtener top performers
        try:
            return self.query_engine.get_top_performers(
                metric=metric, 
                order=order, 
                limit=5,
                filters=filters
            )
        except Exception as e:
            print(f"Error en performance query: {e}")
            return []
    
    def _build_context(self, query: str, semantic_results: List[Dict], 
                      filter_results: List[Dict], ab_analysis: Dict, 
                      performance_results: List[Dict] = None) -> str:
        """Construye contexto enriquecido para OpenAI"""
        
        context = f"""
CONSULTA DEL USUARIO: {query}

{ab_analysis['summary']}

ANÁLISIS POR SEGMENTOS:
"""
        
        # Análisis regional
        if ab_analysis['regional_analysis']:
            context += "\n🌍 ANÁLISIS POR REGIÓN:\n"
            for region, data in ab_analysis['regional_analysis'].items():
                context += f"• {region}: Lift {data['lift']:+.2f}% (Control: {data['control_conversion_rate']:.2f}% → Experimento: {data['experiment_conversion_rate']:.2f}%)\n"
        
        # Análisis por tipo de tienda
        if ab_analysis['store_type_analysis']:
            context += "\n🏪 ANÁLISIS POR TIPO DE TIENDA:\n"
            for store_type, data in ab_analysis['store_type_analysis'].items():
                context += f"• {store_type}: Lift {data['lift']:+.2f}% (Control: {data['control_conversion_rate']:.2f}% → Experimento: {data['experiment_conversion_rate']:.2f}%)\n"
        
        # Resultados de búsqueda semántica
        if semantic_results:
            context += f"\n🔍 DATOS MÁS RELEVANTES (búsqueda semántica):\n"
            for i, result in enumerate(semantic_results[:3], 1):
                context += f"{i}. {result['description']} (Relevancia: {result['score']:.3f})\n"
        
        # Resultados de filtros
        if filter_results:
            context += f"\n📋 DATOS FILTRADOS:\n"
            for i, result in enumerate(filter_results[:3], 1):
                context += f"{i}. {result['description']}\n"
        
        # Resultados de performance (TOP/RANKING)
        if performance_results:
            context += f"\n🏆 TOP PERFORMERS (ranking por métricas):\n"
            for i, result in enumerate(performance_results, 1):
                context += f"{i}. Tienda ID: {result.get('tienda_id', 'N/A')} - "
                context += f"Experimento: {result.get('experimento', 'N/A')} - "
                context += f"Conversión: {result.get('conversion_rate', 0):.2f}% - "
                context += f"Revenue: ${result.get('revenue', 0):,.2f} - "
                context += f"Usuarios: {result.get('usuarios', 0):,}\n"
        
        context += f"\nResponde de manera clara y precisa basándote en estos datos."
        
        return context
    
    def _get_system_prompt(self) -> str:
        """Obtiene el prompt del sistema"""
        return """Eres un analista experto en AB Testing y estadística. 

ESTILO DE RESPUESTA:
- Respuestas CONCISAS y DIRECTAS
- Solo información relevante a la pregunta específica
- Evita análisis extensos no solicitados
- Máximo 3-4 oraciones para preguntas simples
- Análisis detallado solo cuando se solicite explícitamente

FUNCIONES:
1. Responder preguntas específicas sobre datos de AB Testing
2. Interpretar métricas (conversion rate, revenue, lifts)
3. Explicar significancia estadística cuando sea relevante
4. Proporcionar insights cuando se soliciten

REGLAS:
- Basa respuestas solo en datos proporcionados
- Sé específico con números y porcentajes
- Si la pregunta es simple (conteo, datos básicos), responde directamente
- Solo proporciona análisis extenso si se solicita explícitamente"""
    
    def run_console_chat(self):
        """Ejecuta el chatbot en modo consola interactivo"""
        print("\n" + "="*60)
        print("🤖 CHATBOT AB TESTING v2.0 - MODO INTERACTIVO")
        print("="*60)
        print("💡 Ejemplos de preguntas:")
        print("   • ¿Cuál fue el lift en conversiones del experimento?")
        print("   • ¿Cómo se comportaron las tiendas Mall vs Street?")
        print("   • ¿Hay diferencias significativas por región?")
        print("   • ¿Qué región tuvo mejor performance?")
        print("\n📝 Escribe 'salir' para terminar")
        print("-"*60)
        
        while True:
            try:
                user_input = input("\n📊 Tu pregunta: ").strip()
                
                if user_input.lower() in ['salir', 'exit', 'quit', 'q']:
                    print("\n👋 ¡Hasta luego! Gracias por usar el chatbot.")
                    break
                
                if not user_input:
                    print("⚠️  Por favor, escribe una pregunta.")
                    continue
                
                print("\n🔄 Analizando...")
                response = self.generate_response(user_input)
                print(f"\n🤖 Respuesta:\n{response}")
                print("\n" + "-"*60)
                
            except KeyboardInterrupt:
                print("\n\n👋 Chat interrumpido. ¡Hasta luego!")
                break
            except Exception as e:
                print(f"\n❌ Error inesperado: {str(e)}")
                continue

def main():
    """Función principal"""
    try:
        chatbot = ABTestingChatbotV2()
        chatbot.run_console_chat()
    except Exception as e:
        print(f"❌ Error fatal al inicializar chatbot: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()