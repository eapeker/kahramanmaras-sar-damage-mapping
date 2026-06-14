from pystac_client import Client


class STACClient:
    """Copernicus Data Space STAC API ile bağlantı ve görüntü arama."""

    def __init__(self, catalog_url: str = "https://catalogue.dataspace.copernicus.eu/stac"):
        self.catalog_url = catalog_url
        self.client = Client.open(self.catalog_url)

    def search(self, bbox: list, date_range: str, collection: str = "sentinel-1-grd", max_items: int = 10):
        search = self.client.search(
            collections=[collection],
            bbox=bbox,
            datetime=date_range,
            max_items=max_items,
        )
        return list(search.items())


if __name__ == "__main__":
    client = STACClient()
    print("Bağlantı başarılı:", client.client)

    bbox = [36.5, 37.4, 37.2, 37.9]
    items = client.search(bbox=bbox, date_range="2023-02-01/2023-02-15")
    print(f"Bulunan görüntü sayısı: {len(items)}")
    for item in items[:3]:
        print(item.id, item.datetime)
