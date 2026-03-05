from pydantic import BaseModel

class ShipmentFeatures(BaseModel):
    shipping_mode: str
    order_priority: str
    payment_type: str
    profit_per_order: float
    sales_per_customer: float
    order_item_quantity: int