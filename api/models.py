# api/models.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Union 
from datetime import datetime
from enum import Enum 


class ChartType(str, Enum):
    BAR = "bar"
    LINE = "line"
    PIE = "pie"
    SCATTER = "scatter"
    TABLE = "table"
    COMBO = "combo"

class AxisInfo(BaseModel):
    name: str = Field(..., description="اسم العمود المستخدم للبيانات في هذا المحور")
    type: Optional[str] = Field(None, description="نوع بيانات المحور (فئة, رقم, تاريخ, نص)")
    title: Optional[str] = Field(None, description="عنوان المحور المقترح للعرض")

class SeriesInfo(BaseModel):
    name: str = Field(..., description="اسم السلسلة (عادة اسم العمود)")
    color: Optional[str] = Field(None, description="لون السلسلة المقترح (Hex, RGB, أو اسم)")
    type: Optional[ChartType] = Field(None, description="نوع الرسم المقترح لهذه السلسلة (مفيد في COMBO)")


class ChartMetadata(BaseModel):
    timestamp: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"), description="وقت إنشاء البيانات")
    description: str = Field(..., description="وصف الرسم البياني وما يمثله")
    frequency: str = Field(..., description="وتيرة التحديث (يومي, أسبوعي, شهري, ربع سنوي)")
    title: str = Field(..., description="عنوان الرسم البياني المقترح للعرض")
    chart_type: ChartType = Field(..., description="نوع الرسم البياني الأساسي أو العام")
    x_axis: AxisInfo = Field(..., description="معلومات محور X")
    y_axis: AxisInfo = Field(..., description="معلومات محور Y (أو المحور الرئيسي)")
    series: List[SeriesInfo] = Field(..., description="قائمة بالسلاسل (الأعمدة) الممثلة في الرسم")


class ChartData(BaseModel):
    metadata: ChartMetadata = Field(..., description="البيانات الوصفية للرسم")
    data: Union[List[Dict[str, Union[str, float, int, None]]], Dict] = Field(..., description="البيانات الفعلية للرسم بتنسيق JSON مناسب للنوع")



class Product(BaseModel):
    id: int = Field(..., description="معرف المنتج")
    name: str = Field(..., description="اسم المنتج")
    buyPrice: Optional[float] = Field(None, description="سعر الشراء") # جعله اختياريًا إذا كان ممكنًا
    salePrice: Optional[float] = Field(None, description="سعر البيع") # جعله اختياريًا
    quantity: Optional[int] = Field(None, description="الكمية في المخزون") # جعله اختياريًا

class SaleInvoice(BaseModel):
    id: int = Field(..., description="معرف الفاتورة")
    created_at: str = Field(..., description="تاريخ الإنشاء (كنص)") # أو datetime إذا كنت تفضل
    totalPrice: float = Field(..., description="إجمالي المبلغ")

class SaleInvoiceDetail(BaseModel):
    id: int = Field(..., description="المعرف")
    product_id: int = Field(..., description="معرف المنتج")
    invoice_id: int = Field(..., description="معرف فاتورة البيع")
    quantity: int = Field(..., description="الكمية المباعة")
    totalPrice: float = Field(..., description="إجمالي المبلغ")
    buyPrice: Optional[float] = Field(None, description="سعر الشراء (في وقت البيع)") # قد يكون اختياريًا
    created_at: str = Field(..., description="تاريخ الإنشاء (كنص)") # أو datetime

class UploadedData(BaseModel):
    products: List[Product]
    sale_invoices: List[SaleInvoice]
    sale_invoices_details: List[SaleInvoiceDetail]
