# Dinamik Koleksiyon Desteği - Değişiklik Özeti

## Yapılan Değişiklikler (13 Ağustos 2025)

### 1. Dinamik Koleksiyon Yönetimi
- Tüm MCP araçlarına isteğe bağlı `collection_name` parametresi eklendi
- Koleksiyon belirtilmediğinde `default_collection_name` kullanılıyor
- Tek bir MCP sunucusu ile birden fazla koleksiyon yönetilebiliyor

### 2. Response'lara Koleksiyon Bilgisi Eklendi
- `store`: Yanıtta hangi koleksiyona kayıt yapıldığı gösteriliyor
- `find`: Her sonuçta hangi koleksiyondan geldiği belirtiliyor  
- `delete`: Hangi koleksiyondan silindiği bilgisi eklendi
- `get_collection_info`: Koleksiyon adı yanıtta mevcut

### 3. Güncellenen Dosyalar

#### `settings.py`
- `collection_name` → `default_collection_name` olarak değiştirildi
- Açıklama güncellendi: "Varsayılan olarak kullanılacak Qdrant koleksiyonunun adı"

#### `qdrant_memory.py`
- `_ensure_collection(collection_name: str)`: Dinamik koleksiyon oluşturma
- Tüm metodlara `collection_name: str | None = None` parametresi eklendi
- Response formatları koleksiyon bilgisi içerecek şekilde güncellendi

#### `server.py`
- Tüm MCP tool'larına `collection_name` parametresi eklendi
- Store response'u koleksiyon bilgisini gösteriyor
- Lifespan log'ları güncellendi

## Kullanım Örnekleri

```python
# Varsayılan koleksiyona kayıt
qdrant_store("Merhaba dünya")
# Yanıt: "Stored successfully in collection 'mcp_memory' with ID: xxx"

# Belirli bir koleksiyona kayıt
qdrant_store("Clara'nın verisi", collection_name="primary-clara")
# Yanıt: "Stored successfully in collection 'primary-clara' with ID: xxx"

# Belirli koleksiyonda arama
qdrant_find("sorgu", collection_name="primary-clara")
# Yanıt: [{"content": "...", "collection": "primary-clara", ...}]

# Belirli koleksiyondan silme
qdrant_delete("id1,id2", collection_name="primary-clara")
# Yanıt: {"deleted": 2, "ids": ["id1", "id2"], "collection": "primary-clara"}
```

## Başarı Kriterleri ✅
1. ✅ Koleksiyon parametresi verilmeden çağrıldığında varsayılan koleksiyon kullanılıyor
2. ✅ Koleksiyon parametresi ile çağrıldığında belirtilen koleksiyon kullanılıyor
3. ✅ Tüm response'lar hangi koleksiyonda işlem yapıldığını gösteriyor
4. ✅ Tek sunucu ile birden fazla koleksiyon yönetilebiliyor