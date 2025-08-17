"""
Sistema de router de intents para el chatbot AB Testing
Evita cascadas de if/else y facilita el mantenimiento
"""

import re
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum

class IntentType(Enum):
    """Tipos de intents del chatbot"""
    GREETING = "greeting"
    COUNT_QUERY = "count_query" 
    DATA_INFO = "data_info"
    STORE_ID_QUERY = "store_id_query"
    UNKNOWN = "unknown"

@dataclass
class Intent:
    """Definición de un intent"""
    type: IntentType
    patterns: List[str]
    keywords: List[str]
    handler: str  # Nombre del método handler
    priority: int = 1  # Mayor número = mayor prioridad

class IntentRouter:
    """Router de intents para clasificar y dirigir consultas"""
    
    def __init__(self, data_processor, settings):
        self.data_processor = data_processor
        self.settings = settings
        self._define_intents()
    
    def _define_intents(self):
        """Define todos los intents disponibles"""
        self.intents = [
            # SALUDOS - Prioridad alta
            Intent(
                type=IntentType.GREETING,
                patterns=[r"^(hola|hello|hi|buenos días|buenas tardes)[\s\.\!]*$"],
                keywords=["hola", "hello", "hi", "buenos días", "buenas tardes"],
                handler="handle_greeting",
                priority=3
            ),
            
            # CONSULTAS DE CANTIDAD - Prioridad media-alta
            Intent(
                type=IntentType.COUNT_QUERY,
                patterns=[
                    r".*(cuántos|cuántas|cantidad|número|total).*",
                    r".*(how many|count of).*"
                ],
                keywords=["cuántos", "cuántas", "cantidad", "número", "total", "how many", "count"],
                handler="handle_count_query",
                priority=2
            ),
            
            # INFORMACIÓN DE DATOS - Prioridad media
            Intent(
                type=IntentType.DATA_INFO,
                patterns=[
                    r".*(qué datos|qué información|datos disponibles).*",
                    r".*(what data|available data).*"
                ],
                keywords=["qué datos", "qué información", "datos", "información", "tenemos", "disponibles"],
                handler="handle_data_info",
                priority=1
            ),
            
            # CONSULTAS POR ID DE TIENDA - Prioridad alta
            Intent(
                type=IntentType.STORE_ID_QUERY,
                patterns=[
                    r".*T_(Control|Experimento_[ABC])_\d{3}.*",
                    r".*(tienda|store).*id.*",
                    r".*datos.*de.*T_.*"
                ],
                keywords=["T_Control", "T_Experimento", "tienda id", "store id"],
                handler="handle_store_id_query",
                priority=3
            ),
        ]
    
    def classify_intent(self, query: str) -> Intent:
        """Clasifica una consulta y retorna el intent correspondiente"""
        query_lower = query.lower().strip()
        
        # Buscar por patrones primero (más específico)
        for intent in sorted(self.intents, key=lambda x: x.priority, reverse=True):
            for pattern in intent.patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    return intent
        
        # Luego por keywords
        for intent in sorted(self.intents, key=lambda x: x.priority, reverse=True):
            if any(keyword in query_lower for keyword in intent.keywords):
                return intent
        
        # Intent desconocido
        return Intent(
            type=IntentType.UNKNOWN,
            patterns=[],
            keywords=[],
            handler="handle_unknown",
            priority=0
        )
    
    def route_query(self, query: str) -> Optional[str]:
        """Enruta una consulta al handler apropiado"""
        intent = self.classify_intent(query)
        
        # Obtener el handler method
        handler_method = getattr(self, intent.handler, None)
        if handler_method:
            return handler_method(query, intent)
        
        return None
    
    def handle_greeting(self, query: str, intent: Intent) -> str:
        """Handler para saludos"""
        # Solo saludos cortos y simples
        if len(query.strip()) <= 15:
            return "¡Hola! Estoy aquí para ayudarte con el análisis de AB Testing. ¿Qué te gustaría saber sobre los datos?"
        return None  # Saludo complejo, pasarlo a OpenAI
    
    def handle_count_query(self, query: str, intent: Intent) -> str:
        """Handler para consultas de cantidad/conteo"""
        query_lower = query.lower()
        summary = self.data_processor.get_data_summary()
        
        # Consultas combinadas (tiendas Y usuarios)
        if ("tienda" in query_lower or "store" in query_lower) and ("usuario" in query_lower or "user" in query_lower):
            return self._count_stores_and_users(query_lower, summary)
        
        # Router específico para diferentes tipos de conteo
        count_handlers = {
            "usuarios": self._count_users,
            "conversiones": self._count_conversions,
            "tiendas": self._count_stores,
            "regiones": self._count_regions,
            "tipos": self._count_store_types
        }
        
        # Determinar qué tipo de conteo se solicita
        for key, handler in count_handlers.items():
            if key in query_lower or (key == "usuarios" and "user" in query_lower):
                return handler(query_lower, summary)
        
        # Conteo genérico
        return f"El dataset contiene {summary['total_records']} tiendas con {summary['total_users']:,} usuarios y {summary['total_conversions']:,} conversiones en total."
    
    def handle_data_info(self, query: str, intent: Intent) -> str:
        """Handler para información sobre los datos"""
        summary = self.data_processor.get_data_summary()
        return f"""Datos disponibles en el dataset:
• {summary['total_records']} registros de tiendas
• Experimentos: {', '.join(summary['experiments'])} 
• Regiones: {', '.join(summary['regions'])}
• Tipos de tienda: {', '.join(summary['store_types'])}
• Total usuarios: {summary['total_users']:,}
• Total conversiones: {summary['total_conversions']:,}
• Revenue total: ${summary['total_revenue']:,.2f}"""
    
    def handle_store_id_query(self, query: str, intent: Intent) -> str:
        """Handler para consultas específicas por ID de tienda"""
        import re
        
        # Extraer ID de tienda de la consulta
        store_id_pattern = r"T_(Control|Experimento_[ABC])_\d{3}"
        match = re.search(store_id_pattern, query)
        
        if match:
            store_id = match.group(0)
            
            # Buscar en los datos
            store_data = self.data_processor.data[self.data_processor.data['tienda_id'] == store_id]
            
            if not store_data.empty:
                row = store_data.iloc[0]
                return f"""Datos de la tienda {store_id}:
• Experimento: {row['experimento']}
• Región: {row['region']}
• Tipo de tienda: {row['tipo_tienda']}
• Usuarios: {row['usuarios']:,}
• Conversiones: {row['conversiones']:,}
• Revenue: ${row['revenue']:,.2f}
• Conversion Rate: {row['conversion_rate']:.2f}%"""
            else:
                return f"No se encontraron datos para la tienda {store_id}."
        
        return None  # No se encontró ID válido, delegar a OpenAI
    
    def handle_unknown(self, query: str, intent: Intent) -> None:
        """Handler para intents desconocidos - delegar a OpenAI"""
        return None
    
    # Handlers específicos para conteos
    def _count_users(self, query_lower: str, summary: Dict) -> str:
        """Contador específico de usuarios"""
        if "control" in query_lower:
            control_data = self.data_processor.data[self.data_processor.data['experimento'] == 'Control']
            total_users_control = control_data['usuarios'].sum()
            return f"Las tiendas del grupo Control tienen un total de {total_users_control:,} usuarios."
        
        elif any(exp in query_lower for exp in ["experimento a", "experimento_a", "experiment a"]):
            exp_data = self.data_processor.data[self.data_processor.data['experimento'] == 'Experimento_A']
            users = exp_data['usuarios'].sum()
            stores = len(exp_data)
            return f"El Experimento A tiene {stores} tiendas con un total de {users:,} usuarios."
        
        elif any(exp in query_lower for exp in ["experimento", "experiment", "variante"]):
            exp_users = {}
            for exp in ['Experimento_A', 'Experimento_B', 'Experimento_C']:
                exp_data = self.data_processor.data[self.data_processor.data['experimento'] == exp]
                exp_users[exp] = exp_data['usuarios'].sum()
            
            response = "Total de usuarios por experimento:\n"
            for exp, users in exp_users.items():
                response += f"• {exp}: {users:,} usuarios\n"
            return response.strip()
        
        else:
            return f"En total hay {summary['total_users']:,} usuarios en todo el dataset."
    
    def _count_conversions(self, query_lower: str, summary: Dict) -> str:
        """Contador específico de conversiones"""
        if "control" in query_lower:
            control_data = self.data_processor.data[self.data_processor.data['experimento'] == 'Control']
            total_conversions = control_data['conversiones'].sum()
            return f"El grupo Control tiene un total de {total_conversions:,} conversiones."
        else:
            return f"En total hay {summary['total_conversions']:,} conversiones en todo el dataset."
    
    def _count_stores(self, query_lower: str, summary: Dict) -> str:
        """Contador específico de tiendas/registros"""
        if "control" in query_lower:
            control_count = len(self.data_processor.data[self.data_processor.data['experimento'] == 'Control'])
            return f"Hay {control_count} tiendas en el grupo Control."
        
        elif any(exp in query_lower for exp in ["experimento a", "experimento_a", "experiment a"]):
            exp_count = len(self.data_processor.data[self.data_processor.data['experimento'] == 'Experimento_A'])
            return f"El Experimento A tiene {exp_count} tiendas."
        
        elif "experimento" in query_lower:
            exp_counts = self.data_processor.data[self.data_processor.data['experimento'] != 'Control']['experimento'].value_counts()
            response = "Cantidad de tiendas por experimento:\n"
            for exp, count in exp_counts.items():
                response += f"• {exp}: {count} tiendas\n"
            return response.strip()
        
        else:
            return f"El dataset contiene {summary['total_records']} tiendas en total."
    
    def _count_regions(self, query_lower: str, summary: Dict) -> str:
        """Contador específico de regiones"""
        return f"Hay {len(summary['regions'])} regiones en el dataset: {', '.join(summary['regions'])}"
    
    def _count_store_types(self, query_lower: str, summary: Dict) -> str:
        """Contador específico de tipos de tienda"""
        return f"Hay {len(summary['store_types'])} tipos de tienda: {', '.join(summary['store_types'])}"
    
    def _count_stores_and_users(self, query_lower: str, summary: Dict) -> str:
        """Contador combinado de tiendas y usuarios"""
        if "control" in query_lower:
            control_data = self.data_processor.data[self.data_processor.data['experimento'] == 'Control']
            stores = len(control_data)
            users = control_data['usuarios'].sum()
            return f"El grupo Control tiene {stores} tiendas con un total de {users:,} usuarios."
        
        elif any(exp in query_lower for exp in ["experimento a", "experimento_a", "experiment a"]):
            exp_data = self.data_processor.data[self.data_processor.data['experimento'] == 'Experimento_A']
            stores = len(exp_data)
            users = exp_data['usuarios'].sum()
            return f"El Experimento A tiene {stores} tiendas con un total de {users:,} usuarios."
        
        elif any(exp in query_lower for exp in ["experimento b", "experimento_b", "experiment b"]):
            exp_data = self.data_processor.data[self.data_processor.data['experimento'] == 'Experimento_B']
            stores = len(exp_data)
            users = exp_data['usuarios'].sum()
            return f"El Experimento B tiene {stores} tiendas con un total de {users:,} usuarios."
        
        elif any(exp in query_lower for exp in ["experimento c", "experimento_c", "experiment c"]):
            exp_data = self.data_processor.data[self.data_processor.data['experimento'] == 'Experimento_C']
            stores = len(exp_data)
            users = exp_data['usuarios'].sum()
            return f"El Experimento C tiene {stores} tiendas con un total de {users:,} usuarios."
        
        else:
            return f"El dataset contiene {summary['total_records']} tiendas con {summary['total_users']:,} usuarios en total."