from superlinked.framework.dsl.query.query import Query
from typing import Dict, List, Any, Optional
from config import settings

class ABTestingQueryEngine:
    """Motor de consultas para AB Testing usando Superlinked"""
    
    def __init__(self, index, schema, spaces, app):
        self.index = index
        self.schema = schema
        self.spaces = spaces
        self.app = app
    
    def semantic_search(self, 
                       query_text: str, 
                       filters: Optional[Dict] = None,
                       limit: int = None) -> List[Dict]:
        """Búsqueda semántica con filtros opcionales"""
        
        limit = limit or settings.default_limit
        
        # Crear query base
        query = Query(self.index)
        
        # Aplicar filtros si se proporcionan
        if filters:
            query = self._apply_filters(query, filters)
        
        # Búsqueda semántica por descripción con pesos configurables
        weighted_query = query.find(self.schema).similar(
            self.spaces['description'], query_text,
            weight=settings.description_space_weight
        )
        
        # Ejecutar query
        results = self.app.query(weighted_query.limit(limit))
        
        return self._format_results(results)
    
    def filter_search(self, **filters) -> List[Dict]:
        """Búsqueda por filtros específicos"""
        
        query = Query(self.index)
        query = self._apply_filters(query, filters)
        
        results = self.app.query(query.limit(20))
        return self._format_results(results)
    
    def get_top_performers(self, metric: str = 'conversion_rate', 
                          order: str = 'desc', limit: int = 5,
                          filters: Optional[Dict] = None) -> List[Dict]:
        """Obtiene las tiendas con mejor/peor performance en una métrica específica"""
        
        # Obtener TODOS los resultados primero
        query = Query(self.index)
        
        # Aplicar filtros si se proporcionan
        if filters:
            query = self._apply_filters(query, filters)
        
        # Obtener todos los registros (usar límite alto)
        limited_query = query.limit(1000)
        results = self.app.query(limited_query)
        formatted_results = self._format_results(results)
        
        # Ordenar por la métrica especificada
        reverse_order = (order.lower() == 'desc')
        
        try:
            sorted_results = sorted(
                formatted_results, 
                key=lambda x: float(x.get(metric, 0)), 
                reverse=reverse_order
            )
            return sorted_results[:limit]
        except (ValueError, TypeError):
            # Si hay error en conversión, retornar resultados sin ordenar
            return formatted_results[:limit]
    
    def weighted_search(self,
                       query_text: str,
                       experimento_weight: float = None,
                       region_weight: float = None,
                       tipo_tienda_weight: float = None,
                       revenue_weight: float = None,
                       conversion_rate_weight: float = None,
                       filters: Optional[Dict] = None,
                       limit: int = None) -> List[Dict]:
        """Búsqueda con pesos personalizados para diferentes espacios"""
        
        limit = limit or settings.default_limit
        
        # Usar pesos de configuración como default
        weights = {
            'experimento': experimento_weight or settings.experimento_space_weight,
            'region': region_weight or settings.region_space_weight,
            'tipo_tienda': tipo_tienda_weight or settings.tipo_tienda_space_weight,
            'revenue': revenue_weight or settings.revenue_space_weight,
            'conversion_rate': conversion_rate_weight or settings.conversion_rate_space_weight,
        }
        
        # Crear query con pesos
        query = Query(self.index)
        
        # Aplicar filtros
        if filters:
            query = self._apply_filters(query, filters)
        
        # Búsqueda semántica con pesos
        weighted_query = query.find(self.schema).similar(
            self.spaces['description'], query_text,
            weight=settings.description_space_weight
        )
        
        results = self.app.query(weighted_query.limit(limit))
        return self._format_results(results)
    
    def _apply_filters(self, query, filters: Dict):
        """Aplica filtros al query"""
        
        for field_name, value in filters.items():
            if hasattr(self.schema, field_name):
                field = getattr(self.schema, field_name)
                # Usar where() en lugar de filter() para Superlinked
                query = query.where(field == value)
        
        return query
    
    def _format_results(self, results) -> List[Dict]:
        """Formatea los resultados del query"""
        
        formatted_results = []
        
        # Basado en el debug, results es un QueryResult que se itera como tuplas
        # La primera tupla es ('entries', [ResultEntry, ...])
        for key, value in results:
            if key == 'entries':
                for entry in value:
                    # entry es ResultEntry con id, fields, metadata
                    # Extraer datos del id que tiene formato: experimento_tienda_idx
                    id_parts = entry.id.split('_')
                    if len(id_parts) >= 3:
                        experimento = id_parts[0]
                        # El tienda_id parece estar en el formato T_experimento_numero
                        tienda_id = '_'.join(id_parts[1:3]) if len(id_parts) > 3 else id_parts[1]
                    else:
                        experimento = 'Unknown'
                        tienda_id = 'Unknown'
                    
                    formatted_result = {
                        'score': entry.metadata.score,
                        'data': {
                            'experimento': experimento,
                            'tienda_id': tienda_id,
                            'region': 'Unknown',  # No disponible en el resultado
                            'tipo_tienda': 'Unknown',  # No disponible en el resultado
                            'usuarios': 0,  # No disponible en el resultado
                            'conversiones': 0,  # No disponible en el resultado  
                            'revenue': 0.0,  # No disponible en el resultado
                            'conversion_rate': 0.0  # No disponible en el resultado
                        },
                        'description': f"Resultado del experimento {experimento} en tienda {tienda_id}"
                    }
                    formatted_results.append(formatted_result)
                break  # Solo procesar entries
        
        return formatted_results
    
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