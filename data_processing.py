import pandas as pd
import superlinked as sl
from typing import List, Dict
from schema import ABTestingSchema
from config import settings

class ABTestingDataProcessor:
    """Procesador de datos para AB Testing"""
    
    def __init__(self, csv_path: str = None):
        self.csv_path = csv_path or settings.dataset_path
        self.data = self.load_data()
        self.schema = ABTestingSchema()
    
    def load_data(self) -> pd.DataFrame:
        """Carga los datos del CSV"""
        return pd.read_csv(self.csv_path)
    
    def prepare_documents(self) -> List[Dict]:
        """Prepara los documentos para indexar en Superlinked"""
        documents = []
        
        for idx, row in self.data.iterrows():
            # Crear descripción rica para búsqueda semántica
            description = self._create_description(row)
            
            document = {
                'id': f"{row['experimento']}_{row['tienda_id']}_{idx}",
                'experimento': row['experimento'],
                'tienda_id': row['tienda_id'],
                'region': row['region'],
                'tipo_tienda': row['tipo_tienda'],
                'usuarios': int(row['usuarios']),
                'conversiones': int(row['conversiones']),
                'revenue': float(row['revenue']),
                'conversion_rate': float(row['conversion_rate']),
                'description': description
            }
            documents.append(document)
        
        return documents
    
    def _create_description(self, row) -> str:
        """Crea una descripción textual rica para cada registro"""
        description = f"""
        Experimento {row['experimento']} en tienda {row['tienda_id']} 
        ubicada en la región {row['region']} de tipo {row['tipo_tienda']}.
        Resultados: {row['usuarios']} usuarios visitaron la tienda, 
        generando {row['conversiones']} conversiones y un revenue de ${row['revenue']:.2f}.
        La tasa de conversión fue del {row['conversion_rate']}%.
        
        Métricas clave:
        - Usuarios: {row['usuarios']}
        - Conversiones: {row['conversiones']} 
        - Revenue: ${row['revenue']:.2f}
        - Conversion Rate: {row['conversion_rate']}%
        - Ubicación: {row['region']} - {row['tipo_tienda']}
        - Grupo: {row['experimento']}
        """
        return description.strip()
    
    def get_data_summary(self) -> Dict:
        """Obtiene resumen estadístico de los datos"""
        return {
            'total_records': len(self.data),
            'experiments': self.data['experimento'].unique().tolist(),
            'regions': self.data['region'].unique().tolist(),
            'store_types': self.data['tipo_tienda'].unique().tolist(),
            'total_users': self.data['usuarios'].sum(),
            'total_conversions': self.data['conversiones'].sum(),
            'total_revenue': self.data['revenue'].sum(),
            'avg_conversion_rate': self.data['conversion_rate'].mean()
        }