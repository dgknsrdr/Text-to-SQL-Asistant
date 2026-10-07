import json
import random
from pathlib import Path


# ============================================================
# AYARLAR
# ============================================================

SEED = 42

random.seed(SEED)

PROJE_KOKU = Path(__file__).resolve().parent.parent
DATA_KLASORU = PROJE_KOKU / "data"

TRAIN_YOLU = DATA_KLASORU / "train.jsonl"
VALIDATION_YOLU = DATA_KLASORU / "validation.jsonl"
TEST_YOLU = DATA_KLASORU / "test.jsonl"

TRAIN_HEDEF = 4000
VALIDATION_HEDEF = 500
TEST_HEDEF = 500


# ============================================================
# DATABASE BİLGİLERİ
# ============================================================

KATEGORILER = [
    "Salon",
    "Ofis",
    "Yemek Odası",
    "Oturma Grubu"
]

SEHIRLER = [
    "İstanbul",
    "Ankara",
    "İzmir",
    "Bursa"
]

URUN_KELIMELERI = [
    "masa",
    "koltuk",
    "sandalye",
    "kitaplık",
    "ünite"
]


# ============================================================
# TÜRKÇE SAYILAR
# ============================================================

SAYILAR = {
    1: "bir",
    2: "iki",
    3: "üç",
    4: "dört",
    5: "beş",
    6: "altı",
    7: "yedi",
    8: "sekiz",
    9: "dokuz",
    10: "on",
    11: "on bir",
    12: "on iki",
    13: "on üç",
    14: "on dört",
    15: "on beş",
    16: "on altı",
    17: "on yedi",
    18: "on sekiz",
    19: "on dokuz",
    20: "yirmi",
    25: "yirmi beş",
    30: "otuz",
    40: "kırk",
    50: "elli"
}


FIYATLAR = {
    1000: "bin",
    2000: "iki bin",
    3000: "üç bin",
    4000: "dört bin",
    5000: "beş bin",
    6000: "altı bin",
    7000: "yedi bin",
    8000: "sekiz bin",
    9000: "dokuz bin",
    10000: "on bin",
    12000: "on iki bin",
    15000: "on beş bin",
    18000: "on sekiz bin",
    20000: "yirmi bin",
    25000: "yirmi beş bin",
    30000: "otuz bin",
    40000: "kırk bin",
    50000: "elli bin"
}


SIRALAR = {
    1: "birinci",
    2: "ikinci",
    3: "üçüncü",
    4: "dördüncü",
    5: "beşinci",
    6: "altıncı",
    7: "yedinci",
    8: "sekizinci",
    9: "dokuzuncu",
    10: "onuncu"
}


# ============================================================
# YARDIMCI
# ============================================================

def kayit(soru, sql):
    return {
        "soru": soru.strip(),
        "sql": sql.strip()
    }


def turkce_sayi(n):
    return SAYILAR.get(n, str(n))


def turkce_fiyat(n):
    return FIYATLAR.get(n, str(n))


# ============================================================
# TRAIN TEMPLATE'LARI
# ============================================================

TRAIN_TEMPLATES = {

    "stock_lt": [
        "Stoğu {n}'den az olan ürünleri göster.",
        "Stok miktarı {n}'in altında olan ürünleri getir.",
        "Depoda {n} adetten az bulunan ürünleri listele.",
        "Stoğu {word}den az ürünleri göster.",
        "{n} adetten daha az stoğu olan ürünleri getir.",
    ],

    "stock_gt": [
        "Stoğu {n}'den fazla olan ürünleri göster.",
        "Stok miktarı {n}'in üzerinde olan ürünleri getir.",
        "Depoda {n} adetten fazla bulunan ürünleri listele.",
        "Stoğu {word}den fazla ürünleri getir.",
    ],

    "stock_gte": [
        "Stoğu en az {n} olan ürünleri getir.",
        "{n} veya daha fazla stoğu olan ürünleri göster.",
        "{n} ve üzeri stoğa sahip ürünleri listele.",
        "Depoda en az {word} adet bulunan ürünleri getir.",
    ],

    "stock_lte": [
        "Stoğu en fazla {n} olan ürünleri getir.",
        "{n} veya daha az stoğu olan ürünleri göster.",
        "{n} ve altı stoğu olan ürünleri listele.",
        "Depoda en fazla {word} adet bulunan ürünleri getir.",
    ],

    "price_lt": [
        "Fiyatı {n} TL'den düşük ürünleri göster.",
        "{n} TL'den ucuz ürünleri getir.",
        "{word} liradan ucuz ürünleri listele.",
        "Fiyatı {n} liranın altında olanları göster.",
        "{joined}den ucuz ürünleri getir.",
    ],

    "price_gt": [
        "Fiyatı {n} TL'den yüksek ürünleri göster.",
        "{n} TL'den pahalı ürünleri getir.",
        "{word} liradan pahalı ürünleri listele.",
        "Fiyatı {n} liranın üzerinde olanları göster.",
    ],

    "price_gte": [
        "Fiyatı en az {n} TL olan ürünleri getir.",
        "{n} TL ve üzeri ürünleri göster.",
        "{word} lira veya daha pahalı ürünleri listele.",
    ],

    "price_lte": [
        "Fiyatı en fazla {n} TL olan ürünleri getir.",
        "{n} TL ve altındaki ürünleri göster.",
        "{word} lira veya daha ucuz ürünleri listele.",
    ],

    "category": [
        "{category} kategorisindeki ürünleri getir.",
        "{category} kategorisine ait ürünleri göster.",
        "Kategori olarak {category} olan ürünleri listele.",
    ],

    "city": [
        "{city}'da yaşayan müşterileri getir.",
        "{city}'daki müşterileri göster.",
        "Şehri {city} olan müşterileri listele.",
    ],

    "city_name": [
        "{city}'da yaşayan müşterilerin isimlerini getir.",
        "{city}'daki müşteri isimlerini göster.",
    ],

    "price_between": [
        "Fiyatı {a} ile {b} TL arasında olan ürünleri getir.",
        "{a} TL ile {b} TL arasındaki ürünleri göster.",
    ],

    "stock_between": [
        "Stoğu {a} ile {b} arasında olan ürünleri getir.",
        "Stok miktarı {a} ile {b} arasındaki ürünleri göster.",
    ],
}


# ============================================================
# VALIDATION / TEST İÇİN FARKLI DİL KALIPLARI
# ============================================================

EVAL_TEMPLATES = {

    "stock_lt": [
        "Stok seviyesi {n}'in altında kalan ürünler hangileri?",
        "Elde {n}'den daha az bulunan ürünleri getir.",
        "{word} adedin altında stoklanan ürünleri bul.",
    ],

    "stock_gt": [
        "Stok seviyesi {n}'i aşan ürünleri bul.",
        "Elde {n}'den daha çok bulunan ürünleri getir.",
        "{word} adetten fazla kalan ürünleri listele.",
    ],

    "stock_gte": [
        "Minimum {n} adet stoğu bulunan ürünleri getir.",
        "Stok seviyesi {n}'den aşağı olmayan ürünleri bul.",
        "En az {word} tane kalan ürünleri göster.",
    ],

    "stock_lte": [
        "Stok seviyesi maksimum {n} olan ürünleri getir.",
        "{n}'i geçmeyen stok miktarına sahip ürünleri bul.",
        "En çok {word} tane bulunan ürünleri listele.",
    ],

    "price_lt": [
        "Bütçesi {n} TL'nin altında olan ürünleri bul.",
        "{n} lirayı geçmeyen değil, {n} liradan ucuz ürünleri getir.",
        "{word} liranın altında fiyatlandırılmış ürünleri göster.",
    ],

    "price_gt": [
        "{n} liradan daha yüksek fiyatlı ürünler hangileri?",
        "Fiyat etiketi {n} TL'yi aşan ürünleri getir.",
        "{word} liradan pahalı olanları bul.",
    ],

    "price_gte": [
        "Fiyatı minimum {n} lira olan ürünleri bul.",
        "{n} TL'nin altında olmayan ürünleri getir.",
    ],

    "price_lte": [
        "Fiyatı maksimum {n} TL olan ürünleri bul.",
        "{n} lirayı aşmayan ürünleri getir.",
    ],

    "category": [
        "{category} grubuna dahil ürünleri getir.",
        "Ürün grubunda {category} yazan kayıtları göster.",
    ],

    "city": [
        "İkamet şehri {city} olan müşterileri bul.",
        "{city} şehrindeki müşteri kayıtlarını getir.",
    ],

    "city_name": [
        "İkamet yeri {city} olan kişilerin adlarını getir.",
        "{city} şehrinde bulunan müşterilerin isimleri neler?",
    ],

    "price_between": [
        "Fiyat aralığı {a}-{b} TL olan ürünleri bul.",
        "En az {a}, en fazla {b} lira aralığındaki ürünleri getir.",
    ],

    "stock_between": [
        "{a} ile {b} adet arasında stoklanan ürünleri bul.",
        "Stok seviyesi {a}-{b} aralığında olan ürünleri getir.",
    ],
}


# ============================================================
# BASİT ÖRNEK ÜRETİCİ
# ============================================================

def basit_ornek_uret(eval_mode=False):

    templates = EVAL_TEMPLATES if eval_mode else TRAIN_TEMPLATES

    tip = random.choice([
        "stock_lt",
        "stock_gt",
        "stock_gte",
        "stock_lte",
        "price_lt",
        "price_gt",
        "price_gte",
        "price_lte",
        "category",
        "city",
        "city_name",
        "price_between",
        "stock_between"
    ])

    # --------------------------------------------------------
    # STOCK
    # --------------------------------------------------------

    if tip.startswith("stock_") and tip != "stock_between":

        n = random.randint(1, 30)
        word = turkce_sayi(n)

        soru = random.choice(
            templates[tip]
        ).format(
            n=n,
            word=word
        )

        operator = {
            "stock_lt": "<",
            "stock_gt": ">",
            "stock_gte": ">=",
            "stock_lte": "<="
        }[tip]

        sql = (
            f"SELECT * FROM products "
            f"WHERE stock {operator} {n};"
        )

        return kayit(soru, sql)

    # --------------------------------------------------------
    # PRICE
    # --------------------------------------------------------

    if tip.startswith("price_") and tip != "price_between":

        n = random.choice(
            list(FIYATLAR.keys())
        )

        word = turkce_fiyat(n)
        joined = word.replace(" ", "")

        soru = random.choice(
            templates[tip]
        ).format(
            n=n,
            word=word,
            joined=joined
        )

        operator = {
            "price_lt": "<",
            "price_gt": ">",
            "price_gte": ">=",
            "price_lte": "<="
        }[tip]

        sql = (
            f"SELECT * FROM products "
            f"WHERE price {operator} {n};"
        )

        return kayit(soru, sql)

    # --------------------------------------------------------
    # CATEGORY
    # --------------------------------------------------------

    if tip == "category":

        category = random.choice(KATEGORILER)

        soru = random.choice(
            templates["category"]
        ).format(
            category=category
        )

        sql = (
            "SELECT * FROM products "
            f"WHERE category = '{category}';"
        )

        return kayit(soru, sql)

    # --------------------------------------------------------
    # CITY
    # --------------------------------------------------------

    if tip == "city":

        city = random.choice(SEHIRLER)

        soru = random.choice(
            templates["city"]
        ).format(
            city=city
        )

        sql = (
            "SELECT * FROM customers "
            f"WHERE city = '{city}';"
        )

        return kayit(soru, sql)

    if tip == "city_name":

        city = random.choice(SEHIRLER)

        soru = random.choice(
            templates["city_name"]
        ).format(
            city=city
        )

        sql = (
            "SELECT name FROM customers "
            f"WHERE city = '{city}';"
        )

        return kayit(soru, sql)

    # --------------------------------------------------------
    # PRICE BETWEEN
    # --------------------------------------------------------

    if tip == "price_between":

        degerler = sorted(
            random.sample(
                list(FIYATLAR.keys()),
                2
            )
        )

        a, b = degerler

        soru = random.choice(
            templates["price_between"]
        ).format(
            a=a,
            b=b
        )

        sql = (
            "SELECT * FROM products "
            f"WHERE price BETWEEN {a} AND {b};"
        )

        return kayit(soru, sql)

    # --------------------------------------------------------
    # STOCK BETWEEN
    # --------------------------------------------------------

    a = random.randint(1, 15)
    b = random.randint(a + 1, 30)

    soru = random.choice(
        templates["stock_between"]
    ).format(
        a=a,
        b=b
    )

    sql = (
        "SELECT * FROM products "
        f"WHERE stock BETWEEN {a} AND {b};"
    )

    return kayit(soru, sql)


# ============================================================
# ORDER BY / LIMIT / RANK
# ============================================================

def ranking_uret(eval_mode=False):

    alan = random.choice([
        "price",
        "stock"
    ])

    yon = random.choice([
        "ASC",
        "DESC"
    ])

    rank = random.randint(1, 10)

    offset = rank - 1

    rank_word = SIRALAR[rank]

    if alan == "price":

        if yon == "ASC":
            anlam = "ucuz"
        else:
            anlam = "pahalı"

    else:

        if yon == "ASC":
            anlam = "az stoklu"
        else:
            anlam = "çok stoklu"

    if eval_mode:

        templates = [
            "{rank}. sıradaki {anlam} ürünü bul.",
            "{anlam} ürünler arasında {rank_word} olanı getir.",
            "{rank_word} {anlam} ürün hangisi?",
            "{anlam} sıralamasında {rank}. ürünü göster."
        ]

    else:

        templates = [
            "En {anlam} {rank_word} ürünü getir.",
            "{rank_word} en {anlam} ürünü göster.",
            "{anlam} ürünlerden {rank}. sıradakini getir.",
            "Ürünleri {anlam} olacak şekilde sıralayıp {rank_word} ürünü getir."
        ]

    soru = random.choice(
        templates
    ).format(
        rank=rank,
        rank_word=rank_word,
        anlam=anlam
    )

    sql = f"""
SELECT *
FROM products
ORDER BY {alan} {yon}
LIMIT 1 OFFSET {offset};
"""

    return kayit(
        soru,
        sql
    )


# ============================================================
# TOP N
# ============================================================

def top_n_uret(eval_mode=False):

    n = random.randint(2, 10)

    alan = random.choice([
        "price",
        "stock"
    ])

    yon = random.choice([
        "ASC",
        "DESC"
    ])

    if alan == "price":

        anlam = (
            "en ucuz"
            if yon == "ASC"
            else "en pahalı"
        )

    else:

        anlam = (
            "stoğu en az"
            if yon == "ASC"
            else "stoğu en fazla"
        )

    if eval_mode:

        templates = [
            "{anlam} ilk {n} ürünü bul.",
            "{n} tane {anlam} ürünü getir.",
            "{anlam} {n} ürünü listele."
        ]

    else:

        templates = [
            "{anlam} {n} ürünü getir.",
            "{anlam} {n} ürünü göster.",
            "İlk {n} {anlam} ürünü listele."
        ]

    soru = random.choice(
        templates
    ).format(
        anlam=anlam,
        n=n
    )

    sql = f"""
SELECT *
FROM products
ORDER BY {alan} {yon}
LIMIT {n};
"""

    return kayit(soru, sql)


# ============================================================
# AGGREGATE
# ============================================================

def aggregate_uret(eval_mode=False):

    secenekler = [
        (
            "COUNT",
            "*",
            "products",
            "urun_sayisi",
            "ürün sayısı"
        ),
        (
            "AVG",
            "price",
            "products",
            "ortalama_fiyat",
            "ortalama ürün fiyatı"
        ),
        (
            "SUM",
            "stock",
            "products",
            "toplam_stok",
            "toplam stok"
        ),
        (
            "MAX",
            "price",
            "products",
            "en_yuksek_fiyat",
            "en yüksek fiyat"
        ),
        (
            "MIN",
            "price",
            "products",
            "en_dusuk_fiyat",
            "en düşük fiyat"
        ),
        (
            "COUNT",
            "*",
            "customers",
            "musteri_sayisi",
            "müşteri sayısı"
        ),
        (
            "COUNT",
            "*",
            "orders",
            "siparis_sayisi",
            "sipariş sayısı"
        ),
        (
            "SUM",
            "total_price",
            "orders",
            "toplam_tutar",
            "toplam sipariş tutarı"
        ),
        (
            "AVG",
            "total_price",
            "orders",
            "ortalama_tutar",
            "ortalama sipariş tutarı"
        )
    ]

    fonksiyon, kolon, tablo, alias, ifade = random.choice(
        secenekler
    )

    if eval_mode:

        templates = [
            "{ifade} nedir?",
            "Bana {ifade} değerini hesapla.",
            "{ifade} bilgisini getir."
        ]

    else:

        templates = [
            "{ifade} kaç?",
            "{ifade} nedir?",
            "{ifade} değerini göster."
        ]

    soru = random.choice(
        templates
    ).format(
        ifade=ifade
    )

    sql = (
        f"SELECT {fonksiyon}({kolon}) "
        f"AS {alias} FROM {tablo};"
    )

    return kayit(soru, sql)


# ============================================================
# GROUP BY
# ============================================================

def group_by_uret(eval_mode=False):

    tip = random.randint(1, 5)

    if tip == 1:

        sql = """
SELECT category, COUNT(*) AS urun_sayisi
FROM products
GROUP BY category;
"""

        sorular = (
            [
                "Kategori başına ürün adedini hesapla.",
                "Her ürün grubunda kaç kayıt olduğunu getir."
            ]
            if eval_mode
            else
            [
                "Her kategoride kaç ürün var?",
                "Kategori bazında ürün sayılarını göster."
            ]
        )

    elif tip == 2:

        sql = """
SELECT category, AVG(price) AS ortalama_fiyat
FROM products
GROUP BY category;
"""

        sorular = [
            "Kategori bazında ortalama fiyatları getir.",
            "Her kategorinin ortalama ürün fiyatını göster."
        ]

    elif tip == 3:

        sql = """
SELECT category, SUM(stock) AS toplam_stok
FROM products
GROUP BY category;
"""

        sorular = [
            "Her kategorinin toplam stok miktarını göster.",
            "Kategori bazında stok toplamlarını getir."
        ]

    elif tip == 4:

        sql = """
SELECT city, COUNT(*) AS musteri_sayisi
FROM customers
GROUP BY city;
"""

        sorular = [
            "Her şehirde kaç müşteri olduğunu göster.",
            "Şehirlere göre müşteri sayılarını getir."
        ]

    else:

        sql = """
SELECT customer_id, SUM(total_price) AS toplam_tutar
FROM orders
GROUP BY customer_id;
"""

        sorular = [
            "Müşteri id bazında toplam sipariş tutarını göster.",
            "Her müşteri için sipariş toplamını getir."
        ]

    return kayit(
        random.choice(sorular),
        sql
    )


# ============================================================
# HAVING
# ============================================================

def having_uret(eval_mode=False):

    n = random.randint(1, 5)

    tip = random.choice([
        "category_count",
        "city_count"
    ])

    if tip == "category_count":

        soru = random.choice([
            f"{n}'den fazla ürünü olan kategorileri göster.",
            f"Ürün sayısı {n}'i geçen kategorileri getir.",
            f"En az {n + 1} ürünü bulunan kategorileri listele."
        ])

        sql = f"""
SELECT category, COUNT(*) AS urun_sayisi
FROM products
GROUP BY category
HAVING COUNT(*) > {n};
"""

    else:

        soru = random.choice([
            f"{n}'den fazla müşterisi olan şehirleri göster.",
            f"Müşteri sayısı {n}'i geçen şehirleri getir.",
            f"En az {n + 1} müşterisi bulunan şehirleri listele."
        ])

        sql = f"""
SELECT city, COUNT(*) AS musteri_sayisi
FROM customers
GROUP BY city
HAVING COUNT(*) > {n};
"""

    return kayit(soru, sql)


# ============================================================
# AND / OR
# ============================================================

def coklu_kosul_uret(eval_mode=False):

    stok = random.randint(1, 20)

    fiyat = random.choice(
        list(FIYATLAR.keys())
    )

    operator_tip = random.choice([
        "AND",
        "OR"
    ])

    if operator_tip == "AND":

        soru = random.choice([
            f"Stoğu en az {stok} ve fiyatı {fiyat} TL'den düşük ürünleri göster.",
            f"Stok miktarı {stok} veya üzeri olup {fiyat} liradan ucuz ürünleri getir.",
            f"Hem stoğu {stok}'den az olmayan hem de fiyatı {fiyat} TL'nin altında olan ürünleri listele."
        ])

        sql = f"""
SELECT *
FROM products
WHERE stock >= {stok}
AND price < {fiyat};
"""

    else:

        soru = random.choice([
            f"Stoğu {stok}'den az veya fiyatı {fiyat} TL'den fazla ürünleri göster.",
            f"Stok miktarı {stok}'in altında ya da fiyatı {fiyat} lirayı geçen ürünleri getir."
        ])

        sql = f"""
SELECT *
FROM products
WHERE stock < {stok}
OR price > {fiyat};
"""

    return kayit(soru, sql)


# ============================================================
# CATEGORY + PRICE/STOCK
# ============================================================

def kategori_kosul_uret(eval_mode=False):

    kategori = random.choice(KATEGORILER)

    fiyat = random.choice(
        list(FIYATLAR.keys())
    )

    stok = random.randint(1, 20)

    tip = random.choice([
        1,
        2,
        3
    ])

    if tip == 1:

        soru = (
            f"{kategori} kategorisinde fiyatı "
            f"{fiyat} TL'den düşük ürünleri getir."
        )

        sql = f"""
SELECT *
FROM products
WHERE category = '{kategori}'
AND price < {fiyat};
"""

    elif tip == 2:

        soru = (
            f"{kategori} kategorisinde stoğu "
            f"en az {stok} olan ürünleri göster."
        )

        sql = f"""
SELECT *
FROM products
WHERE category = '{kategori}'
AND stock >= {stok};
"""

    else:

        soru = (
            f"{kategori} kategorisinde stoğu en az {stok} "
            f"ve fiyatı {fiyat} TL'nin altında olan ürünleri getir."
        )

        sql = f"""
SELECT *
FROM products
WHERE category = '{kategori}'
AND stock >= {stok}
AND price < {fiyat};
"""

    return kayit(soru, sql)


# ============================================================
# JOIN
# ============================================================

def join_uret(eval_mode=False):

    tip = random.randint(
        1,
        7
    )

    if tip == 1:

        soru = random.choice([
            "Siparişleri müşteri isimleriyle göster.",
            "Her siparişin sahibinin adını getir.",
            "Sipariş kayıtlarını müşteri adlarıyla listele."
        ])

        sql = """
SELECT o.id, c.name, o.order_date, o.total_price
FROM orders o
JOIN customers c
ON o.customer_id = c.id;
"""

    elif tip == 2:

        soru = random.choice([
            "Her müşterinin kaç sipariş verdiğini göster.",
            "Müşteri bazında sipariş sayılarını getir."
        ])

        sql = """
SELECT c.name, COUNT(o.id) AS siparis_sayisi
FROM customers c
LEFT JOIN orders o
ON c.id = o.customer_id
GROUP BY c.id, c.name;
"""

    elif tip == 3:

        soru = random.choice([
            "Müşterilerin toplam harcamalarını göster.",
            "Her müşterinin toplam ne kadar harcadığını getir."
        ])

        sql = """
SELECT c.name, SUM(o.total_price) AS toplam_harcama
FROM customers c
JOIN orders o
ON c.id = o.customer_id
GROUP BY c.id, c.name;
"""

    elif tip == 4:

        soru = random.choice([
            "En fazla para harcayan müşteriyi getir.",
            "Toplam harcaması en yüksek müşteriyi göster."
        ])

        sql = """
SELECT c.name, SUM(o.total_price) AS toplam_harcama
FROM customers c
JOIN orders o
ON c.id = o.customer_id
GROUP BY c.id, c.name
ORDER BY toplam_harcama DESC
LIMIT 1;
"""

    elif tip == 5:

        rank = random.randint(
            2,
            5
        )

        rank_word = SIRALAR[rank]

        soru = random.choice([
            f"En fazla harcama yapan {rank_word} müşteriyi getir.",
            f"Toplam harcamada {rank}. sıradaki müşteriyi göster."
        ])

        sql = f"""
SELECT c.name, SUM(o.total_price) AS toplam_harcama
FROM customers c
JOIN orders o
ON c.id = o.customer_id
GROUP BY c.id, c.name
ORDER BY toplam_harcama DESC
LIMIT 1 OFFSET {rank - 1};
"""

    elif tip == 6:

        soru = random.choice([
            "En fazla sipariş veren müşteriyi getir.",
            "Sipariş sayısı en yüksek müşteriyi göster."
        ])

        sql = """
SELECT c.name, COUNT(o.id) AS siparis_sayisi
FROM customers c
JOIN orders o
ON c.id = o.customer_id
GROUP BY c.id, c.name
ORDER BY siparis_sayisi DESC
LIMIT 1;
"""

    else:

        rank = random.randint(
            2,
            5
        )

        soru = (
            f"Sipariş sayısında {rank}. sıradaki müşteriyi getir."
        )

        sql = f"""
SELECT c.name, COUNT(o.id) AS siparis_sayisi
FROM customers c
JOIN orders o
ON c.id = o.customer_id
GROUP BY c.id, c.name
ORDER BY siparis_sayisi DESC
LIMIT 1 OFFSET {rank - 1};
"""

    return kayit(soru, sql)


# ============================================================
# SUBQUERY
# ============================================================

def subquery_uret(eval_mode=False):

    tip = random.randint(
        1,
        6
    )

    if tip == 1:

        soru = "Ortalama fiyattan pahalı ürünleri getir."

        sql = """
SELECT *
FROM products
WHERE price > (
    SELECT AVG(price)
    FROM products
);
"""

    elif tip == 2:

        soru = "Ortalama fiyattan ucuz ürünleri getir."

        sql = """
SELECT *
FROM products
WHERE price < (
    SELECT AVG(price)
    FROM products
);
"""

    elif tip == 3:

        soru = "Ortalama stoktan fazla stoğu olan ürünleri getir."

        sql = """
SELECT *
FROM products
WHERE stock > (
    SELECT AVG(stock)
    FROM products
);
"""

    elif tip == 4:

        soru = "En pahalı ürün veya ürünleri getir."

        sql = """
SELECT *
FROM products
WHERE price = (
    SELECT MAX(price)
    FROM products
);
"""

    elif tip == 5:

        soru = "En ucuz ürün veya ürünleri getir."

        sql = """
SELECT *
FROM products
WHERE price = (
    SELECT MIN(price)
    FROM products
);
"""

    else:

        soru = "Stoğu en düşük ürün veya ürünleri getir."

        sql = """
SELECT *
FROM products
WHERE stock = (
    SELECT MIN(stock)
    FROM products
);
"""

    return kayit(soru, sql)


# ============================================================
# ILIKE
# ============================================================

def text_search_uret(eval_mode=False):

    kelime = random.choice(
        URUN_KELIMELERI
    )

    tip = random.randint(
        1,
        3
    )

    if tip == 1:

        soru = (
            f"Adında {kelime} geçen ürünleri getir."
        )

        sql = f"""
SELECT *
FROM products
WHERE name ILIKE '%{kelime}%';
"""

    elif tip == 2:

        harf = random.choice(
            list("mksto")
        )

        soru = (
            f"Adı {harf} harfiyle başlayan ürünleri getir."
        )

        sql = f"""
SELECT *
FROM products
WHERE name ILIKE '{harf}%';
"""

    else:

        harf = random.choice(
            list("aktes")
        )

        soru = (
            f"Adı {harf} harfiyle biten ürünleri getir."
        )

        sql = f"""
SELECT *
FROM products
WHERE name ILIKE '%{harf}';
"""

    return kayit(
        soru,
        sql
    )


# ============================================================
# IN / NOT IN
# ============================================================

def in_uret(eval_mode=False):

    tip = random.choice([
        "city",
        "category"
    ])

    not_in = random.choice([
        True,
        False
    ])

    if tip == "city":

        a, b = random.sample(
            SEHIRLER,
            2
        )

        if not_in:

            soru = (
                f"{a} ve {b} dışındaki müşterileri getir."
            )

            sql = f"""
SELECT *
FROM customers
WHERE city NOT IN ('{a}', '{b}');
"""

        else:

            soru = (
                f"{a} veya {b}'de yaşayan müşterileri getir."
            )

            sql = f"""
SELECT *
FROM customers
WHERE city IN ('{a}', '{b}');
"""

    else:

        a, b = random.sample(
            KATEGORILER,
            2
        )

        if not_in:

            soru = (
                f"{a} ve {b} dışındaki kategorilere ait ürünleri getir."
            )

            sql = f"""
SELECT *
FROM products
WHERE category NOT IN ('{a}', '{b}');
"""

        else:

            soru = (
                f"{a} veya {b} kategorisindeki ürünleri getir."
            )

            sql = f"""
SELECT *
FROM products
WHERE category IN ('{a}', '{b}');
"""

    return kayit(soru, sql)


# ============================================================
# DATE
# ============================================================

def tarih_uret(eval_mode=False):

    yil = 2026

    ay = random.randint(
        1,
        12
    )

    sonraki_yil = yil
    sonraki_ay = ay + 1

    if sonraki_ay == 13:
        sonraki_ay = 1
        sonraki_yil += 1

    baslangic = f"{yil}-{ay:02d}-01"
    bitis = f"{sonraki_yil}-{sonraki_ay:02d}-01"

    tip = random.randint(
        1,
        3
    )

    if tip == 1:

        soru = (
            f"2026 yılının {ay}. ayındaki siparişleri getir."
        )

        sql = f"""
SELECT *
FROM orders
WHERE order_date >= '{baslangic}'
AND order_date < '{bitis}';
"""

    elif tip == 2:

        soru = (
            f"2026 yılının {ay}. ayında kaç sipariş verilmiş?"
        )

        sql = f"""
SELECT COUNT(*) AS siparis_sayisi
FROM orders
WHERE order_date >= '{baslangic}'
AND order_date < '{bitis}';
"""

    else:

        soru = (
            f"2026 yılının {ay}. ayındaki toplam sipariş tutarı nedir?"
        )

        sql = f"""
SELECT SUM(total_price) AS toplam_tutar
FROM orders
WHERE order_date >= '{baslangic}'
AND order_date < '{bitis}';
"""

    return kayit(soru, sql)


# ============================================================
# DISTINCT
# ============================================================

def distinct_uret(eval_mode=False):

    if random.random() < 0.5:

        return kayit(
            random.choice([
                "Farklı ürün kategorilerini getir.",
                "Ürün kategorilerini tekrarsız göster.",
                "Kaç çeşit kategori olduğunu görmek için kategorileri tekil getir."
            ]),
            "SELECT DISTINCT category FROM products;"
        )

    return kayit(
        random.choice([
            "Müşterilerin bulunduğu farklı şehirleri getir.",
            "Şehirleri tekrarsız göster.",
            "Müşterilerin bulunduğu tekil şehir listesini getir."
        ]),
        "SELECT DISTINCT city FROM customers;"
    )


# ============================================================
# NULL
# ============================================================

def null_uret(eval_mode=False):

    kolon = random.choice([
        "category",
        "price",
        "stock"
    ])

    null_mu = random.choice([
        True,
        False
    ])

    isimler = {
        "category": "kategori bilgisi",
        "price": "fiyat bilgisi",
        "stock": "stok bilgisi"
    }

    if null_mu:

        soru = (
            f"{isimler[kolon]} girilmemiş ürünleri getir."
        )

        sql = (
            f"SELECT * FROM products "
            f"WHERE {kolon} IS NULL;"
        )

    else:

        soru = (
            f"{isimler[kolon]} bulunan ürünleri getir."
        )

        sql = (
            f"SELECT * FROM products "
            f"WHERE {kolon} IS NOT NULL;"
        )

    return kayit(soru, sql)


# ============================================================
# GENEL ÜRETİCİ
# ============================================================

URETICILER = [
    basit_ornek_uret,
    ranking_uret,
    top_n_uret,
    aggregate_uret,
    group_by_uret,
    having_uret,
    coklu_kosul_uret,
    kategori_kosul_uret,
    join_uret,
    subquery_uret,
    text_search_uret,
    in_uret,
    tarih_uret,
    distinct_uret,
    null_uret
]


def dataset_uret(
    hedef,
    mevcut_sorular=None,
    eval_mode=False
):

    if mevcut_sorular is None:
        mevcut_sorular = set()

    sonuc = []

    deneme = 0

    maksimum_deneme = hedef * 100

    while (
        len(sonuc) < hedef
        and deneme < maksimum_deneme
    ):

        deneme += 1

        uretici = random.choice(
            URETICILER
        )

        ornek = uretici(
            eval_mode=eval_mode
        )

        soru_key = (
            ornek["soru"]
            .strip()
            .lower()
        )

        if soru_key in mevcut_sorular:
            continue

        mevcut_sorular.add(
            soru_key
        )

        sonuc.append(
            ornek
        )

    if len(sonuc) < hedef:

        raise RuntimeError(
            f"{hedef} benzersiz örnek üretilemedi. "
            f"Sadece {len(sonuc)} üretildi."
        )

    return sonuc


# ============================================================
# DATASETLER
# ============================================================

tum_sorular = set()


print(
    "Train dataset oluşturuluyor..."
)

train_data = dataset_uret(
    TRAIN_HEDEF,
    mevcut_sorular=tum_sorular,
    eval_mode=False
)


print(
    "Validation dataset oluşturuluyor..."
)

validation_data = dataset_uret(
    VALIDATION_HEDEF,
    mevcut_sorular=tum_sorular,
    eval_mode=True
)


print(
    "Test dataset oluşturuluyor..."
)

test_data = dataset_uret(
    TEST_HEDEF,
    mevcut_sorular=tum_sorular,
    eval_mode=True
)


# ============================================================
# KARIŞTIR
# ============================================================

random.shuffle(
    train_data
)

random.shuffle(
    validation_data
)

random.shuffle(
    test_data
)


# ============================================================
# JSONL YAZ
# ============================================================

DATA_KLASORU.mkdir(
    parents=True,
    exist_ok=True
)


def jsonl_yaz(
    yol,
    veri
):

    with open(
        yol,
        "w",
        encoding="utf-8"
    ) as dosya:

        for ornek in veri:

            dosya.write(
                json.dumps(
                    ornek,
                    ensure_ascii=False
                )
                + "\n"
            )


jsonl_yaz(
    TRAIN_YOLU,
    train_data
)

jsonl_yaz(
    VALIDATION_YOLU,
    validation_data
)

jsonl_yaz(
    TEST_YOLU,
    test_data
)


# ============================================================
# SONUÇ
# ============================================================

print()
print("=" * 60)
print("DATASET OLUŞTURULDU")
print("=" * 60)

print(
    "Train:",
    len(train_data)
)

print(
    "Validation:",
    len(validation_data)
)

print(
    "Test:",
    len(test_data)
)

print(
    "Toplam:",
    (
        len(train_data)
        + len(validation_data)
        + len(test_data)
    )
)

print()
print("Dosyalar:")
print(TRAIN_YOLU)
print(VALIDATION_YOLU)
print(TEST_YOLU)