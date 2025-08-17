import pandas as pd
import numpy as np
from typing import Dict, List, Any
import os
from dotenv import load_dotenv
from openai import OpenAI
from scipy import stats
from superlinked.framework.dsl.space.categorical_similarity_space import CategoricalSimilaritySpace
from superlinked.framework.dsl.space.number_space import NumberSpace
from superlinked.framework.dsl.space.text_similarity_space import TextSimilaritySpace
from superlinked.framework.dsl.index.index import Index
from superlinked.framework.dsl.query.query import Query
from superlinked.framework.dsl.storage.in_memory_vector_database import InMemoryVectorDatabase
from superlinked.framework.dsl.executor.in_memory.in_memory_executor import InMemoryExecutor
from superlinked.framework.common.schema.id_schema_object import IdSchemaObject
from superlinked.framework.common.schema.schema import schema
from superlinked.framework.common.schema.schema_object import String, Integer, Float

load_dotenv()

@schema
class ABTestingSchema:
    id: IdSchemaObject
    experimento: String
    tienda_id: String 
    region: String
    tipo_tienda: String
    usuarios: Integer
    conversiones: Integer
    revenue: Float
    conversion_rate: Float
    description: String

class ABTestingChatbot:
    def __init__(self, csv_file_path: str):
        self.csv_file_path = csv_file_path
        self.data = self.load_data()
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.schema = ABTestingSchema()
        self.vector_index = self.create_superlinked_index()
        
    def load_data(self) -> pd.DataFrame:
        """Carga y prepara los datos del CSV"""
        df = pd.read_csv(self.csv_file_path)
        return df
    
    def create_superlinked_index(self):
        """Crea índice vectorial multi-atributo usando Superlinked"""
        # Definir espacios vectoriales para diferentes atributos
        experimento_space = CategoricalSimilaritySpace(
            self.schema.experimento, 
            categories=["Control", "Experimento_A"],
            negative_filter=0.8
        )
        
        region_space = CategoricalSimilaritySpace(
            self.schema.region,
            categories=["Este", "Norte", "Sur", "Oeste"],
            negative_filter=0.7
        )
        
        tipo_tienda_space = CategoricalSimilaritySpace(
            self.schema.tipo_tienda,
            categories=["Mall", "Street", "Outlet"],
            negative_filter=0.7
        )
        
        # Espacios numéricos para métricas
        usuarios_space = NumberSpace(
            self.schema.usuarios,
            min_value=0,
            max_value=500,
            mode_norm=lambda x: x / 500.0
        )
        
        conversiones_space = NumberSpace(
            self.schema.conversiones,
            min_value=0,
            max_value=50,
            mode_norm=lambda x: x / 50.0
        )
        
        revenue_space = NumberSpace(
            self.schema.revenue,
            min_value=0,
            max_value=1000,
            mode_norm=lambda x: x / 1000.0
        )
        
        conversion_rate_space = NumberSpace(
            self.schema.conversion_rate,
            min_value=0,
            max_value=20,
            mode_norm=lambda x: x / 20.0
        )
        
        # Espacio de texto para descripciones
        description_space = TextSimilaritySpace(
            self.schema.description,
            model="text-embedding-3-small"
        )
        
        # Crear índice combinando todos los espacios
        index = Index([
            experimento_space,
            region_space, 
            tipo_tienda_space,
            usuarios_space,
            conversiones_space,
            revenue_space,
            conversion_rate_space,
            description_space
        ])
        
        # Configurar base de datos vectorial y executor
        vector_db = InMemoryVectorDatabase()
        executor = InMemoryExecutor(vector_db)
        
        # Preparar y cargar datos
        documents = []
        for idx, row in self.data.iterrows():
            description = f"""
            Experimento {row['experimento']} en tienda {row['tienda_id']} 
            ubicada en región {row['region']} de tipo {row['tipo_tienda']}.
            Métricas: {row['usuarios']} usuarios, {row['conversiones']} conversiones,
            ${row['revenue']:.2f} revenue, {row['conversion_rate']}% conversion rate.
            """
            
            doc = {
                self.schema.id: str(idx),
                self.schema.experimento: row['experimento'],
                self.schema.tienda_id: row['tienda_id'],
                self.schema.region: row['region'],
                self.schema.tipo_tienda: row['tipo_tienda'],
                self.schema.usuarios: int(row['usuarios']),
                self.schema.conversiones: int(row['conversiones']),
                self.schema.revenue: float(row['revenue']),
                self.schema.conversion_rate: float(row['conversion_rate']),
                self.schema.description: description.strip()
            }
            documents.append(doc)
        
        # Indexar documentos
        executor.put(documents, index)
        
        return {
            'index': index,
            'executor': executor,
            'schema': self.schema
        }
    
    def analyze_ab_test_results(self) -> Dict[str, Any]:
        """Analiza los resultados del AB Testing"""
        control_data = self.data[self.data['experimento'] == 'Control']
        experiment_data = self.data[self.data['experimento'] == 'Experimento_A']
        
        # Métricas agregadas
        control_metrics = {
            'total_usuarios': control_data['usuarios'].sum(),
            'total_conversiones': control_data['conversiones'].sum(),
            'total_revenue': control_data['revenue'].sum(),
            'avg_conversion_rate': control_data['conversion_rate'].mean()
        }
        
        experiment_metrics = {
            'total_usuarios': experiment_data['usuarios'].sum(),
            'total_conversiones': experiment_data['conversiones'].sum(),
            'total_revenue': experiment_data['revenue'].sum(),
            'avg_conversion_rate': experiment_data['conversion_rate'].mean()
        }
        
        # Calcular lifts
        conversion_lift = ((experiment_metrics['avg_conversion_rate'] - control_metrics['avg_conversion_rate']) / control_metrics['avg_conversion_rate']) * 100
        revenue_lift = ((experiment_metrics['total_revenue'] - control_metrics['total_revenue']) / control_metrics['total_revenue']) * 100
        
        # Test de significancia estadística
        control_rates = control_data['conversion_rate'].values
        experiment_rates = experiment_data['conversion_rate'].values
        t_stat, p_value = stats.ttest_ind(control_rates, experiment_rates)
        
        return {
            'control': control_metrics,
            'experiment': experiment_metrics,
            'conversion_lift': conversion_lift,
            'revenue_lift': revenue_lift,
            'statistical_significance': {
                't_statistic': t_stat,
                'p_value': p_value,
                'is_significant': p_value < 0.05
            }
        }
    
    def find_similar_data(self, query: str, filters: Dict = None, top_k: int = 5) -> List[Dict]:
        """Busca datos similares usando Superlinked multi-attribute indexing"""
        # Crear query usando Superlinked
        query_obj = Query(self.vector_index['index'])
        
        # Añadir filtros si se proporcionan
        if filters:
            for field, value in filters.items():
                if hasattr(self.schema, field):
                    query_obj = query_obj.filter(getattr(self.schema, field) == value)
        
        # Buscar por similitud de texto en la descripción
        query_obj = query_obj.find(self.schema.description).similar(query, limit=top_k)
        
        # Ejecutar query
        results = self.vector_index['executor'].query(query_obj)
        
        # Formatear resultados
        formatted_results = []
        for result in results:
            formatted_results.append({
                'similarity': result.score,
                'data': {
                    'experimento': result.document[self.schema.experimento],
                    'tienda_id': result.document[self.schema.tienda_id],
                    'region': result.document[self.schema.region],
                    'tipo_tienda': result.document[self.schema.tipo_tienda],
                    'usuarios': result.document[self.schema.usuarios],
                    'conversiones': result.document[self.schema.conversiones],
                    'revenue': result.document[self.schema.revenue],
                    'conversion_rate': result.document[self.schema.conversion_rate]
                },
                'description': result.document[self.schema.description]
            })
        
        return formatted_results
    
    def query_by_attributes(self, **filters) -> List[Dict]:
        """Busca datos por atributos específicos usando Superlinked"""
        query_obj = Query(self.vector_index['index'])
        
        # Aplicar filtros múltiples
        for field, value in filters.items():
            if hasattr(self.schema, field):
                query_obj = query_obj.filter(getattr(self.schema, field) == value)
        
        # Ejecutar query
        results = self.vector_index['executor'].query(query_obj.limit(20))
        
        # Formatear resultados
        formatted_results = []
        for result in results:
            formatted_results.append({
                'data': {
                    'experimento': result.document[self.schema.experimento],
                    'tienda_id': result.document[self.schema.tienda_id],
                    'region': result.document[self.schema.region],
                    'tipo_tienda': result.document[self.schema.tipo_tienda],
                    'usuarios': result.document[self.schema.usuarios],
                    'conversiones': result.document[self.schema.conversiones],
                    'revenue': result.document[self.schema.revenue],
                    'conversion_rate': result.document[self.schema.conversion_rate]
                },
                'description': result.document[self.schema.description]
            })
        
        return formatted_results
    
    def generate_response(self, user_query: str) -> str:
        """Genera respuesta usando OpenAI con contexto de los datos de Superlinked"""
        # Detectar posibles filtros en la consulta
        filters = self.extract_filters_from_query(user_query)
        
        # Buscar datos relevantes usando Superlinked
        relevant_data = self.find_similar_data(user_query, filters)
        
        # También buscar por atributos específicos si hay filtros
        attribute_data = []
        if filters:
            attribute_data = self.query_by_attributes(**filters)
        
        # Obtener análisis general
        ab_analysis = self.analyze_ab_test_results()
        
        # Crear contexto para OpenAI
        context = f"""
        Eres un analista experto en AB Testing. Tienes acceso a los siguientes datos:
        
        ANÁLISIS GENERAL DEL EXPERIMENTO:
        - Control: {ab_analysis['control']['total_usuarios']} usuarios, {ab_analysis['control']['total_conversiones']} conversiones, ${ab_analysis['control']['total_revenue']:.2f} revenue
        - Experimento A: {ab_analysis['experiment']['total_usuarios']} usuarios, {ab_analysis['experiment']['total_conversiones']} conversiones, ${ab_analysis['experiment']['total_revenue']:.2f} revenue
        - Lift en conversión: {ab_analysis['conversion_lift']:.2f}%
        - Lift en revenue: {ab_analysis['revenue_lift']:.2f}%
        - Significancia estadística: {'Sí' if ab_analysis['statistical_significance']['is_significant'] else 'No'} (p-value: {ab_analysis['statistical_significance']['p_value']:.4f})
        
        DATOS MÁS RELEVANTES PARA LA CONSULTA (usando búsqueda vectorial):
        """
        
        for item in relevant_data:
            context += f"\n{item['description']} (Similitud: {item['similarity']:.3f})\n"
        
        if attribute_data:
            context += "\nDATOS FILTRADOS POR ATRIBUTOS:\n"
            for item in attribute_data[:5]:  # Limitar a 5 resultados
                context += f"\n{item['description']}\n"
        
        context += f"""
        
        Responde a la siguiente pregunta de manera clara y precisa, usando los datos proporcionados:
        PREGUNTA: {user_query}
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Eres un analista experto en AB Testing. Proporciona respuestas claras, precisas y basadas en datos."},
                {"role": "user", "content": context}
            ],
            temperature=0.3
        )
        
        return response.choices[0].message.content
    
    def extract_filters_from_query(self, query: str) -> Dict:
        """Extrae filtros potenciales de la consulta del usuario"""
        filters = {}
        query_lower = query.lower()
        
        # Detectar experimento
        if 'control' in query_lower:
            filters['experimento'] = 'Control'
        elif 'experimento' in query_lower or 'variante' in query_lower:
            filters['experimento'] = 'Experimento_A'
        
        # Detectar región
        regions = ['este', 'norte', 'sur', 'oeste']
        for region in regions:
            if region in query_lower:
                filters['region'] = region.capitalize()
                break
        
        # Detectar tipo de tienda
        if 'mall' in query_lower:
            filters['tipo_tienda'] = 'Mall'
        elif 'street' in query_lower or 'calle' in query_lower:
            filters['tipo_tienda'] = 'Street'
        elif 'outlet' in query_lower:
            filters['tipo_tienda'] = 'Outlet'
        
        return filters
    
    def run_console_chat(self):
        """Ejecuta el chatbot en consola"""
        print("🤖 Chatbot de Análisis AB Testing")
        print("=" * 50)
        print("Datos cargados exitosamente!")
        print(f"Total de registros: {len(self.data)}")
        print("Escribe 'salir' para terminar.\n")
        
        while True:
            user_input = input("📊 Tu pregunta: ").strip()
            
            if user_input.lower() in ['salir', 'exit', 'quit']:
                print("¡Hasta luego! 👋")
                break
            
            if not user_input:
                continue
            
            print("\n🔄 Procesando...")
            try:
                response = self.generate_response(user_input)
                print(f"\n🤖 Respuesta:\n{response}\n")
                print("-" * 50)
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                print("-" * 50)

def main():
    chatbot = ABTestingChatbot('tiendas_detalle.csv')
    chatbot.run_console_chat()

if __name__ == "__main__":
    main()