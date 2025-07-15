import { map } from "./map";
import { Vector as VectorLayer } from 'ol/layer';
import { Vector as VectorSource } from 'ol/source';
import { Draw, Snap } from 'ol/interaction';
import GeoJSON from 'ol/format/GeoJSON';
import { v4 as uuidv4 } from 'uuid';
import { fromLonLat } from "ol/proj";
import { Polygon } from "ol/geom";
import { Feature } from "ol";
import { Circle as CircleStyle, Fill, Stroke, Style } from 'ol/style';

// 1. Layer und Source wie gehabt definieren
const photoArSource = new VectorSource();
export const photoArLayer = new VectorLayer({
  title: "drawarea",
  source: photoArSource,
  style: new Style({
    fill: new Fill({ color: 'rgba(121, 253, 145, 0.4)' }),
    stroke: new Stroke({ color: '#ffcc33', width: 2 }),
    image: new CircleStyle({
      radius: 7,
      fill: new Fill({ color: '#ffcc33' }),
      stroke: new Stroke({ color: '#f39c12', width: 1 })
    })
  })
});

// 2. JSON laden und Polygone erzeugen
fetch('/js/footprints.json')
  .then(response => {
    if (!response.ok) {
      throw new Error("Failed to load photoareas.json");
    }
    return response.json();
  })
  .then(areaArray => {
    areaArray.forEach(entry => {
      // 3. Koordinaten aus JSON (lon, lat) transformieren
      const transformedCoords = entry.footprint.map(([lon, lat]) =>
        fromLonLat([lon, lat])
      );
      // 4. Polygon-Feature bauen (geschlossenes Koordinaten-Loop)
      const polygon = new Polygon([[
        ...transformedCoords,
        transformedCoords[0]  // sicherstellen, dass das Polygon geschlossen ist
      ]]);
      const feature = new Feature(polygon);
      feature.setId(entry.filename);  // optional: ID setzen
      photoArSource.addFeature(feature);
    });
  })
  .catch(error => {
    console.error("Error loading JSON file:", error);
  });
