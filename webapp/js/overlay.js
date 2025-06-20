// Note the explicit .js extensions if you're in an ES-module environment.
import Overlay from 'ol/Overlay.js';
import KMLFormat from 'ol/format/KML.js';
import { map } from './map';
export function initDownloadPopup(map) {
  const container = document.getElementById('popup');
  const content   = document.getElementById('popup-content');
  const closer    = document.getElementById('popup-closer');

  // 1. Create and add the Overlay
  const popupOverlay = new Overlay({
    element: container,
    autoPan: true,
    autoPanAnimation: { duration: 250 }
  });
  map.addOverlay(popupOverlay);

  // 2. Close button
  closer.addEventListener('click', () => {
    popupOverlay.setPosition(undefined);
    closer.blur();
    return false;
  });

  // 3. Click handler
  map.on('singleclick', (evt) => {
    const feature = map.forEachFeatureAtPixel(evt.pixel, f => f);
    if (!feature) {
      popupOverlay.setPosition(undefined);
      return;
    }

    const geom = feature.getGeometry();
    if (!geom || !geom.getType().startsWith('Polygon')) {
      popupOverlay.setPosition(undefined);
      return;
    }

    // 4. Serialize to KML in lat/lon (EPSG:4326)
    const kmlFormatter = new KMLFormat({
      // your map is probably in EPSG:3857 (WebMercator) or EPSG:4326
      featureProjection: map.getView().getProjection(),  
      dataProjection: 'EPSG:4326'
    });
    const kmlText = kmlFormatter.writeFeatures([feature], {
    featureProjection: map.getView().getProjection(),
    dataProjection: 'EPSG:4326'
  });
    console.log(feature)
    // 5. Build download link
    const blob = new Blob([kmlText], {
      type: 'application/vnd.google-earth.kml+xml'
    });
    const url   = URL.createObjectURL(blob);
    const title = feature.get('title') || 'area';

    // 6. Populate and show the popup
    content.innerHTML = `
      <p><strong>${title}</strong></p>
      <a href="${url}" download="${title}.kml">Download KML</a>
    `;
    popupOverlay.setPosition(evt.coordinate);
  });
}