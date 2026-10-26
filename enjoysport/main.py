
import csv

# Membaca data training dari file CSV

def baca_data(nama_file):
    data = []

    with open(nama_file, "r", newline="", encoding="utf-8-sig") as file:
        pembaca = csv.reader(file)
        header = next(pembaca)

        header = [kolom.strip() for kolom in header]

        for baris in pembaca:
            if not baris:
                continue

            baris = [nilai.strip() for nilai in baris]
            atribut = baris[:-1]
            label = baris[-1]
            data.append((atribut, label))

    return header[:-1], data


# Algoritma Find-S
def find_s(data):
    h = None

    print("\n=== PROSES FIND-S ===")

    for atribut, label in data:
        if label == "Yes":
            if h is None:
                h = atribut.copy()
            else:
                for i in range(len(h)):
                    if h[i] != atribut[i]:
                        h[i] = "?"

            print("Contoh positif:", atribut)
            print("Hipotesis h:", h)

    return h


# Memeriksa apakah hipotesis cocok dengan data
def cocok(h, atribut):
    for i in range(len(h)):
        if h[i] != "?" and h[i] != atribut[i]:
            return False
    return True


# Memeriksa apakah hipotesis h1 lebih umum
# atau sama dengan h2
def lebih_umum(h1, h2):
    for i in range(len(h1)):
        if h1[i] != "?" and h1[i] != h2[i]:
            return False
    return True


# Menghapus hipotesis yang duplikat
def unik(daftar):
    hasil = []
    for h in daftar:
        if h not in hasil:
            hasil.append(h)
    return hasil


# Algoritma Candidate-Elimination
def candidate_elimination(nama_atribut, data):
    jumlah = len(nama_atribut)
    S = [[None] * jumlah]
    G = [["?"] * jumlah]

    domain = [[] for _ in range(jumlah)]

    for atribut, label in data:
        for i in range(jumlah):
            if atribut[i] not in domain[i]:
                domain[i].append(atribut[i])

    print("\n=== PROSES CANDIDATE-ELIMINATION ===")

    for atribut, label in data:
        if label == "Yes":
            # Hapus hipotesis G yang tidak cocok
            G = [g for g in G if cocok(g, atribut)]

            # Generalisasi S agar cocok dengan contoh positif
            S_baru = []

            for s in S:
                h = s.copy()

                for i in range(jumlah):
                    if h[i] is None:
                        h[i] = atribut[i]
                    elif h[i] != atribut[i]:
                        h[i] = "?"

                S_baru.append(h)

            S = S_baru

        else:
            # Hapus S yang mencakup contoh negatif
            S = [s for s in S if not cocok(s, atribut)]

            # Spesialisasi hipotesis G
            G_baru = []

            for g in G:
                if not cocok(g, atribut):
                    G_baru.append(g)
                    continue

                for i in range(jumlah):
                    if g[i] == "?":
                        for nilai in domain[i]:
                            if nilai != atribut[i]:
                                h = g.copy()
                                h[i] = nilai

                                if any(
                                    lebih_umum(h, s) for s in S
                                ):
                                    G_baru.append(h)

            G = unik(G_baru)

        print("Data:", atribut, label)
        print("S =", S)
        print("G =", G)
        print()

    return S, G


# Program utama
nama_atribut, data = baca_data("enjoysport.csv")

print("Atribut:", nama_atribut)
print("Jumlah data training:", len(data))

# Menjalankan Find-S
h_akhir = find_s(data)

print("\nHipotesis akhir Find-S:", h_akhir)

# Meminta data hari baru
print("\n=== PREDIKSI HARI BARU ===")
hari_baru = []

for atribut in nama_atribut:
    nilai = input("Masukkan " + atribut + ": ")
    hari_baru.append(nilai)

if cocok(h_akhir, hari_baru):
    print("Prediksi Find-S: Yes")
else:
    print("Prediksi Find-S: No")

# Menjalankan Candidate-Elimination
candidate_elimination(nama_atribut, data)
