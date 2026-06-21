import logging
from typing import Optional, Sequence, Tuple, Iterator, Union

from pystac_client import Client
from pystac import Item

logger = logging.getLogger(__name__)


class STACClient:
    """Copernicus Data Space STAC API ile bağlantı ve görüntü arama.

    Özellikler:
    - Dışarıdan `Client` enjeksiyonu (test kolaylığı).
    - Basit bbox ve tarih aralığı doğrulaması.
    - İstisna yakalama ve logging.
    - `search(... )` bir iterator döndürür; sonuçlar bellekte toplanmaz.
    """

    def __init__(
        self,
        catalog_url: str = "https://catalogue.dataspace.copernicus.eu/stac",
        client: Optional[Client] = None,
    ):
        self.catalog_url = catalog_url
        if client is not None:
            self.client = client
        else:
            try:
                self.client = Client.open(self.catalog_url)
            except Exception as exc:  # pragma: no cover - surface network/config errors
                logger.exception("Failed to open STAC catalog at %s", self.catalog_url)
                raise

    @staticmethod
    def _validate_bbox(bbox: Sequence[float]) -> Tuple[float, float, float, float]:
        if len(bbox) != 4:
            raise ValueError("bbox must be a sequence of four floats: [minx, miny, maxx, maxy]")
        minx, miny, maxx, maxy = tuple(bbox)
        if not (minx < maxx and miny < maxy):
            raise ValueError("bbox values invalid: require minx < maxx and miny < maxy")
        return minx, miny, maxx, maxy

    def search(
        self,
        bbox: Sequence[float],
        date_range: Union[str, Tuple[str, str]],
        collection: str = "sentinel-1-grd",
        max_items: Optional[int] = None,
    ) -> Iterator[Item]:
        """STAC araması yapar ve `Item` iterator döndürür.

        Args:
            bbox: [minx, miny, maxx, maxy]
            date_range: 'YYYY-MM-DD/YYYY-MM-DD' veya ('YYYY-MM-DD','YYYY-MM-DD')
            collection: koleksiyon adı
            max_items: maksimum döndürülecek öğe sayısı (None = sınırsız)

        Returns:
            Iterator over `pystac.Item` objects.
        """
        minx, miny, maxx, maxy = self._validate_bbox(bbox)

        if isinstance(date_range, tuple):
            datetime = f"{date_range[0]}/{date_range[1]}"
        else:
            datetime = date_range

        try:
            search = self.client.search(
                collections=[collection],
                bbox=[minx, miny, maxx, maxy],
                datetime=datetime,
                max_items=max_items,
            )
        except Exception:
            logger.exception("STAC search failed for bbox=%s datetime=%s collection=%s", bbox, datetime, collection)
            raise

        # `search.items()` returns a lazy iterator; kullanıcı isterse list() ile toplayabilir.
        return search.items()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    client = STACClient()
    logger.info("Bağlantı başarılı: %s", client.client)

    # Doğru sıra: [minx, miny, maxx, maxy]
    bbox = [36.5, 37.2, 37.9, 37.9]
    items_iter = client.search(bbox=bbox, date_range=("2023-02-01", "2023-02-15"))

    # örnek olarak ilk 3 öğeyi alalım
    items = []
    for i, it in enumerate(items_iter):
        if i >= 3:
            break
        items.append(it)

    logger.info("Örnek alınan öğe sayısı: %d", len(items))
    for item in items:
        logger.info("%s %s", item.id, getattr(item, "datetime", "-"))
