from decimal import Decimal

from database.connection import (
    veritabani_baglantisi
)


YASAKLI_KOMUTLAR = [
    "delete",
    "update",
    "insert",
    "drop",
    "alter",
    "truncate",
    "create",
    "grant",
    "revoke"
]


def sorgu_guvenli_mi(sql: str) -> bool:

    sql_temiz = sql.strip().lower()

    if not sql_temiz.startswith("select"):
        return False

    for komut in YASAKLI_KOMUTLAR:

        if komut in sql_temiz:
            return False

    return True


def sql_calistir(sql: str):

    if not sorgu_guvenli_mi(sql):

        raise ValueError(
            "Güvenlik nedeniyle sadece SELECT sorguları çalıştırılabilir."
        )

    with veritabani_baglantisi() as baglanti:

        with baglanti.cursor() as cursor:

            cursor.execute(sql)

            kolonlar = [
                aciklama.name
                for aciklama
                in cursor.description
            ]

            satirlar = cursor.fetchall()

    sonuclar = []

    for satir in satirlar:

        kayit = {}

        for kolon, deger in zip(
            kolonlar,
            satir
        ):

            if isinstance(
                deger,
                Decimal
            ):
                deger = float(deger)

            kayit[kolon] = deger

        sonuclar.append(
            kayit
        )

    return sonuclar