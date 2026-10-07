from database.connection import veritabani_baglantisi


with veritabani_baglantisi() as baglanti:
    with baglanti.cursor() as cursor:
        cursor.execute("SELECT * FROM products;")

        for satir in cursor.fetchall():
            print(satir)