from app.text_to_sql import TextToSQLModel
from database.sql_tool import sql_calistir


# Model sadece bir kere yüklenir
text_to_sql_model = TextToSQLModel()


def sonuc_metni_olustur(sonuclar):

    if not sonuclar:

        return "Bu sorguya uygun kayıt bulunamadı."

    if len(sonuclar) == 1:

        kayit = sonuclar[0]

        parcalar = []

        for kolon, deger in kayit.items():

            parcalar.append(
                f"{kolon}: {deger}"
            )

        return ", ".join(parcalar)


    cevap = (
        f"Toplam {len(sonuclar)} kayıt bulundu:\n"
    )

    for index, kayit in enumerate(
        sonuclar,
        start=1
    ):

        degerler = []

        for kolon, deger in kayit.items():

            degerler.append(
                f"{kolon}: {deger}"
            )

        cevap += (
            f"\n{index}. "
            + ", ".join(degerler)
        )

    return cevap


while True:

    print("\n" + "=" * 60)

    soru = input(
        "\nŞirket AI'ına sorun: "
    )

    if not soru.strip():
        continue

    if soru.lower() == "exit":
        break


    try:

        # -----------------------------------------
        # 1. Türkçe soru → SQL
        # -----------------------------------------

        sql = text_to_sql_model.sql_uret(
            soru
        )


        print(
            "\nÜretilen SQL:"
        )

        print(
            sql
        )


        # -----------------------------------------
        # 2. SQL → PostgreSQL
        # -----------------------------------------

        sonuclar = sql_calistir(
            sql
        )


        # -----------------------------------------
        # 3. PostgreSQL sonucu → kullanıcı cevabı
        # -----------------------------------------

        cevap = sonuc_metni_olustur(
            sonuclar
        )


        print(
            "\nAI Cevabı:"
        )

        print(
            cevap
        )


    except Exception as hata:

        print(
            "\nBir hata oluştu:"
        )

        print(
            hata
        )