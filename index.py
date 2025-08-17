from superlinked.framework.dsl.space.categorical_similarity_space import CategoricalSimilaritySpace
from superlinked.framework.dsl.space.number_space import NumberSpace, Mode
from superlinked.framework.dsl.space.text_similarity_space import TextSimilaritySpace
from superlinked.framework.dsl.index.index import Index
from schema import ABTestingSchema
from config import settings

def create_ab_testing_index():
    """Crea el índice de Superlinked para datos de AB Testing"""
    
    schema = ABTestingSchema()
    
    # Definir espacios vectoriales con pesos configurables
    experimento_space = CategoricalSimilaritySpace(
        schema.experimento,
        categories=["Control", "Experimento_A"],
        negative_filter=0.8
    )
    
    region_space = CategoricalSimilaritySpace(
        schema.region,
        categories=["Este", "Norte", "Sur", "Oeste"],
        negative_filter=0.7
    )
    
    tipo_tienda_space = CategoricalSimilaritySpace(
        schema.tipo_tienda,
        categories=["Mall", "Street", "Outlet"], 
        negative_filter=0.7
    )
    
    # Espacios numéricos para métricas
    usuarios_space = NumberSpace(
        schema.usuarios,
        min_value=0,
        max_value=500,
        mode=Mode.SIMILAR
    )
    
    conversiones_space = NumberSpace(
        schema.conversiones,
        min_value=0,
        max_value=50,
        mode=Mode.SIMILAR
    )
    
    revenue_space = NumberSpace(
        schema.revenue,
        min_value=0,
        max_value=1000,
        mode=Mode.SIMILAR
    )
    
    conversion_rate_space = NumberSpace(
        schema.conversion_rate,
        min_value=0,
        max_value=20,
        mode=Mode.SIMILAR
    )
    
    # Espacio de texto para descripciones
    description_space = TextSimilaritySpace(
        schema.description,
        model=settings.embedding_model
    )
    
    # Crear índice con todos los espacios vectoriales
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
    
    # Crear diccionario de espacios para facilitar el acceso
    spaces = {
        'experimento': experimento_space,
        'region': region_space,
        'tipo_tienda': tipo_tienda_space,
        'usuarios': usuarios_space,
        'conversiones': conversiones_space,
        'revenue': revenue_space,
        'conversion_rate': conversion_rate_space,
        'description': description_space
    }
    
    return index, schema, spaces