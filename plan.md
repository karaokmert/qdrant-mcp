# **İŞ PLANI: Qdrant MCP Sunucusunu Geliştirme**

**Proje Adı:** Merkezi Hafıza Servisi: Dinamik Koleksiyon Yönetimi Yeteneği

**Tarih:** 13 Ağustos 2025

**Sorumlu Agent:** \[Geliştirici Agent Adı\]

**Gözden Geçiren:** Mert Karaok, Clara

### **1\. Amaç**

Bu projenin temel amacı, karaokmert/qdrant-mcp isimli forkladığımız Python tabanlı MCP sunucusunu, tek bir COLLECTION\_NAME'e bağlı kalmak yerine, her bir API isteğinde dinamik olarak farklı koleksiyonlarla çalışabilecek şekilde revize etmektir. Bu değişiklik, AI ekosistemimizin bellek sorunlarını kalıcı olarak çözecek ve ölçeklenebilir bir hafıza altyapısı kurmamızı sağlayacaktır.

### **2\. Mevcut Durum ve Problem Tanımı**

Forklanan qdrant-mcp sunucusu, başlangıçta ortam değişkenlerinden okuduğu tek ve sabit bir COLLECTION\_NAME ile çalışmaktadır. Bu mimari, her bir agent koleksiyonu için ayrı bir sunucu süreci başlatmamızı gerektirir, bu da sistem kaynaklarının (özellikle bellek) verimsiz kullanılmasına ve yönetilemez bir yapıya yol açar.

### **3\. Hedeflenen Durum**

Proje tamamlandığında, qdrant-mcp sunucusu aşağıdaki yeteneklere sahip olacaktır:

* Tek bir Python süreci olarak çalışacak.  
* Başlangıçta bir DEFAULT\_COLLECTION\_NAME (varsayılan koleksiyon) alacak.  
* store, find, delete gibi tüm araçlarında, isteğe bağlı bir collection\_name parametresi kabul edecek.  
* Eğer istekte collection\_name parametresi belirtilirse, o koleksiyonda işlem yapacak.  
* Eğer collection\_name parametresi belirtilmezse, DEFAULT\_COLLECTION\_NAME'de işlem yapacak.

### **4\. Geliştirme Adımları ve Görevler**

#### **Görev 1: Hafıza Katmanını Esnekleştirme**

**Dosya:** src/qdrant\_mcp/qdrant\_memory.py

* **1.1: \_ensure\_collection Fonksiyonunu Güncelle:**  
  * Fonksiyonun collection\_name: str parametresi almasını sağla.  
  * Fonksiyonun, parametre olarak gelen koleksiyonun varlığını kontrol etmesini ve yoksa, embedding\_provider'dan aldığı doğru dimensions ile oluşturmasını sağla.  
  * \_collection\_initialized bayrağını kaldır, çünkü artık her koleksiyon dinamik olarak kontrol edilecek.  
* **1.2: Ana Fonksiyonları Güncelle (store, find, delete, get\_collection\_info):**  
  * Her bir fonksiyona, isteğe bağlı collection\_name: str | None \= None parametresini ekle.  
  * Fonksiyonun içinde, target\_collection \= collection\_name or self.settings.collection\_name mantığıyla hedef koleksiyonu belirle.  
  * Fonksiyonun geri kalanındaki tüm self.settings.collection\_name kullanımlarını target\_collection ile değiştir.

#### **Görev 2: Sunucu Araçlarını Güçlendirme**

**Dosya:** src/qdrant\_mcp/server.py

* **2.1: Tüm MCP Araçlarını Güncelle (qdrant\_store, qdrant\_find, qdrant\_delete, qdrant\_collection\_info):**  
  * Her bir @mcp.tool() fonksiyon tanımına, isteğe bağlı collection\_name: str | None \= None parametresini ekle.  
  * Fonksiyonun içinde, qdrant\_client'ın ilgili metodunu çağırırken, bu yeni collection\_name parametresini de alt katmana ilet.

#### **Görev 3: Ayarları Netleştirme**

**Dosya:** src/qdrant\_mcp/settings.py

* **3.1: collection\_name Alanını Güncelle:**  
  * collection\_name alanının adını default\_collection\_name olarak değiştir.  
  * Açıklamasını (description) "Varsayılan olarak kullanılacak Qdrant koleksiyonunun adı" olarak güncelle. Bu, kodun niyetini daha net hale getirecektir.

### **5\. Başarı Kriterleri ve Test Senaryoları**

Geliştirme tamamlandıktan sonra aşağıdaki senaryoların başarıyla çalıştığı teyit edilmelidir:

1. qdrant\_store aracı, collection\_name parametresi **verilmeden** çağrıldığında, veriyi ayarlarda belirtilen default\_collection\_name'e kaydetmelidir.  
2. qdrant\_store aracı, collection\_name: "primary-clara" parametresi **verilerek** çağrıldığında, veriyi primary-clara koleksiyonuna kaydetmelidir.  
3. qdrant\_find aracı, farklı collection\_name parametreleri ile çağrıldığında, sadece o koleksiyon içinden doğru sonuçları getirmelidir.