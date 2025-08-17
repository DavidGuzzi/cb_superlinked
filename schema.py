from superlinked.framework.common.schema.schema import schema
from superlinked.framework.common.schema.schema_object import String, Integer, Float
from superlinked.framework.common.schema.id_schema_object import IdField

@schema
class ABTestingSchema:
    """Schema para datos de AB Testing usando Superlinked"""
    
    id: IdField
    experimento: String
    tienda_id: String
    region: String
    tipo_tienda: String
    usuarios: Integer
    conversiones: Integer
    revenue: Float
    conversion_rate: Float
    description: String