# Enterprise AI Copilot

Enterprise AI Copilot, şirket içi veritabanı ve doküman kaynaklarını tek bir arayüz üzerinden sorgulamayı amaçlayan bir yapay zeka uygulamasıdır.

Sistem, kullanıcının sorusunu analiz ederek isteğin veritabanı sorgusu mu yoksa doküman tabanlı bir soru mu olduğunu belirler.

- Veritabanı soruları için fine-tuned Qwen tabanlı Text-to-SQL modeli kullanılır.
- Doküman soruları için RAG tabanlı retrieval sistemi kullanılır.
- Backend FastAPI ile geliştirilmiştir.
- Veritabanı olarak PostgreSQL kullanılmıştır.
- Kullanıcı etkileşimi için web tabanlı bir frontend hazırlanmıştır.

---

## Proje Mimarisi

```text
                Kullanıcı
                   │
                   ▼
              Web Arayüzü
                   │
                   ▼
                FastAPI
                   │
                   ▼
                 Router
              /          \
             /            \
       Database           Document
          │                  │
          ▼                  ▼
 Fine-tuned Qwen           RAG
          │                  │
          ▼                  ▼
         SQL            Embedding Search
          │                  │
          ▼                  ▼
     PostgreSQL        Company Documents
          │                  │
          └────────┬─────────┘
                   ▼
              Kullanıcı Cevabı

# Enterprise AI Copilot

Enterprise AI Copilot, şirket içi veritabanı ve doküman kaynaklarını tek bir arayüz üzerinden sorgulamayı amaçlayan bir yapay zeka uygulamasıdır.

Sistem, kullanıcının sorusunu analiz ederek isteğin veritabanı sorgusu mu yoksa doküman tabanlı bir soru mu olduğunu belirler.

- Veritabanı soruları için fine-tuned Qwen tabanlı Text-to-SQL modeli kullanılır.
- Doküman soruları için RAG tabanlı retrieval sistemi kullanılır.
- Backend FastAPI ile geliştirilmiştir.
- Veritabanı olarak PostgreSQL kullanılmıştır.
- Kullanıcı etkileşimi için web tabanlı bir frontend hazırlanmıştır.

---

## Proje Mimarisi

```text
                Kullanıcı
                   │
                   ▼
              Web Arayüzü
                   │
                   ▼
                FastAPI
                   │
                   ▼
                 Router
              /          \
             /            \
       Database           Document
          │                  │
          ▼                  ▼
 Fine-tuned Qwen           RAG
          │                  │
          ▼                  ▼
         SQL            Embedding Search
          │                  │
          ▼                  ▼
     PostgreSQL        Company Documents
          │                  │
          └────────┬─────────┘
                   ▼
              Kullanıcı Cevabı

Özellikler
- Türkçe doğal dil sorularını SQL sorgularına dönüştürme
- PostgreSQL veritabanı ile entegrasyon
- Fine-tuned Qwen Text-to-SQL modeli
- LoRA / QLoRA tabanlı fine-tuning
- RAG tabanlı doküman sorgulama
- Embedding tabanlı semantic search
- SQL ve doküman sorularını ayıran router sistemi
- FastAPI backend
- Web tabanlı kullanıcı arayüzü
- SQL sorgu sonucu görüntüleme
- Üretilen SQL sorgusunu arayüzde gösterme
- .env tabanlı güvenli veritabanı konfigürasyonu
Kullanılan Teknolojiler
AI / Machine Learning
- Python
- PyTorch
- Hugging Face Transformers
- Qwen
- PEFT
- LoRA / QLoRA
- TRL
- BitsAndBytes
- Sentence Transformers
Backend
- FastAPI
- Uvicorn
- Pydantic
Database
- PostgreSQL
- Psycopg
Frontend
- HTML
- CSS
- JavaScript
Text-to-SQL Modeli
Projede küçük bir Qwen modeli Text-to-SQL görevi için fine-tune edilmiştir.
Model girdisi:
### Veritabanı Şeması

products(id,name,category,price,stock)
customers(id,name,city)
orders(id,customer_id,order_date,total_price)

İlişki:
orders.customer_id -> customers.id

### Soru

Toplam harcaması en yüksek ikinci müşteriyi getir

### SQL

Model çıktısı:
SELECT c.name, SUM(o.total_price) AS toplam_harcama
FROM customers c
JOIN orders o
ON c.id = o.customer_id
GROUP BY c.id, c.name
ORDER BY toplam_harcama DESC
LIMIT 1 OFFSET 1;

Fine-Tuning
Model PEFT ve LoRA kullanılarak fine-tune edilmiştir.
Eğitim veri seti:
Train:      4000 örnek
Validation: 500 örnek
Test:       500 örnek

Fine-tuning sırasında yaklaşık 6.4 milyon parametre trainable olarak kullanılmıştır.
Model eğitiminde kullanılan temel yaklaşım:
Base Qwen Model
      +
4-bit Quantization
      +
LoRA Adapter
      =
Fine-tuned Text-to-SQL Model

Validation sonucunda en iyi checkpoint otomatik olarak seçilmiştir.
Desteklenen SQL Yapıları
Eğitim veri setinde farklı SQL kalıpları kullanılmıştır.
SELECT
WHERE
AND / OR
BETWEEN
IN / NOT IN
ORDER BY
ASC / DESC
LIMIT
OFFSET
COUNT
SUM
AVG
MIN
MAX
DISTINCT
GROUP BY
HAVING
JOIN
LEFT JOIN
ILIKE
Subquery
Ranking
Date filtering
NULL kontrolü

RAG Sistemi
Doküman tabanlı sorular Text-to-SQL modeline gönderilmez.
Örneğin:
Teşhir ürünlerinde iade süresi kaç gün?

Router bu soruyu doküman sorgusu olarak algılar.
Akış:
Soru
  ↓
Embedding
  ↓
Semantic Search
  ↓
En alakalı doküman parçaları
  ↓
Cevap

Örnek şirket politikası dokümanları:
İade Politikası
Teslimat Politikası
Garanti Politikası

Router
Sistem kullanıcı sorularını iki farklı route'a yönlendirir.
Database
Örnek:
En pahalı ürünü getir

Akış:
Router
  ↓
Text-to-SQL
  ↓
PostgreSQL
  ↓
Sonuç

Document
Örnek:
İkinci el ürünlerde garanti var mı?

Akış:
Router
  ↓
RAG
  ↓
Company Documents
  ↓
Sonuç

Veritabanı Yapısı
Products
products(
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    category VARCHAR(100),
    price NUMERIC(10,2),
    stock INTEGER
)

Customers
customers(
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    city VARCHAR(100)
)

Orders
orders(
    id SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(id),
    order_date DATE,
    total_price NUMERIC(10,2)
)

Proje Yapısı
LLM/
│
├── app/
│   ├── __init__.py
│   ├── api.py
│   ├── main.py
│   ├── rag_service.py
│   ├── router.py
│   └── text_to_sql.py
│
├── database/
│   ├── __init__.py
│   ├── connection.py
│   └── sql_tool.py
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
│
├── documents/
│   └── company_policy.txt
│
├── data/
│   ├── train.jsonl
│   ├── validation.jsonl
│   ├── test.jsonl
│   ├── train_formatted.jsonl
│   ├── validation_formatted.jsonl
│   └── test_formatted.jsonl
│
├── training/
│   ├── generate_dataset.py
│   ├── prepare_dataset.py
│   └── train.py
│
├── evaluation/
│   ├── benchmark.py
│   ├── benchmark_execution.py
│   ├── evaluate.py
│   └── test_model.py
│
├── models/
│   └── qwen_text_to_sql_lora/
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md

Kurulum
Repository'yi klonlayın:
git clone <repository-url>
cd LLM

Virtual environment oluşturun:
python -m venv .venv

Windows:
.venv\Scripts\activate

Gerekli paketleri yükleyin:
pip install -r requirements.txt

Environment Variables
Proje kök dizininde .env dosyası oluşturun.
DB_HOST=localhost
DB_PORT=5432
DB_NAME=enterprise_ai
DB_USER=postgres
DB_PASSWORD=your_password

Gerçek .env dosyası GitHub'a yüklenmemelidir.
Bunun yerine .env.example kullanılabilir.
Uygulamayı Çalıştırma
Proje kök dizininde:
python -m uvicorn app.api:app --reload

Tarayıcı:
http://127.0.0.1:8000

API sağlık kontrolü:
http://127.0.0.1:8000/health

Örnek Sorular
Database Soruları
En pahalı ürünü getir

Toplam harcaması en yüksek ikinci müşteriyi getir

Stoğu 5'ten fazla olan ürünleri listele

Ortalama stok miktarının üstünde stoğu olan ürünleri fiyatına göre sırala

Document Soruları
Teşhir ürünlerinde iade süresi kaç gün?

İkinci el ürünlerde garanti var mı?

İstanbul içi teslimat ne kadar sürüyor?

Örnek API Response
Database Route
{
  "success": true,
  "route": "database",
  "question": "En pahalı ürünü getir",
  "sql": "SELECT * FROM products ORDER BY price DESC LIMIT 1;",
  "answer": "name: Modern Koltuk, price: 18500",
  "data": []
}

Document Route
{
  "success": true,
  "route": "document",
  "question": "Teşhir ürünlerinde iade süresi kaç gün?",
  "sql": null,
  "answer": "Teşhir ürünlerinde iade süresi 7 gündür.",
  "data": []
}

Model Limitasyonları
Text-to-SQL modeli küçük ölçekli ve proje amaçlı fine-tune edilmiştir.
Model:
- bazı karmaşık SQL kombinasyonlarında hata üretebilir,
- doğal dil ifadelerini zaman zaman yanlış yorumlayabilir,
- gerçek production sistemlerde ek SQL validation ve güvenlik katmanları gerektirir.
Bu proje, çalışan bir AI Copilot mimarisini uçtan uca göstermek amacıyla geliştirilmiştir.
Gelecek Geliştirmeler
Planlanan geliştirmeler:
- Text-to-SQL veri setinin genişletilmesi
- Daha karmaşık SQL kombinasyonlarının desteklenmesi
- Router sisteminin LLM tabanlı hale getirilmesi
- Vector database entegrasyonu
- Daha gelişmiş RAG pipeline
- Conversation history
- Kullanıcı authentication sistemi
- Role-based database access
- SQL execution sandbox
- Docker desteği
- Cloud deployment
- Streaming response
- Daha gelişmiş frontend
Projenin Amacı
Bu proje ile aşağıdaki konuların tek bir uygulamada birleştirilmesi hedeflenmiştir:
LLM Fine-Tuning
Text-to-SQL
RAG
Embeddings
PostgreSQL
FastAPI
Web Development
AI System Design

Amaç sadece bir modeli eğitmek değil, eğitilen modeli gerçek bir uygulama içerisinde veritabanı ve doküman kaynaklarıyla birlikte kullanılabilir hale getirmektir.