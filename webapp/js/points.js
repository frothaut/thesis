import { map } from "./map";
import { Vector as VectorLayer } from 'ol/layer';
import { Vector, Vector as VectorSource } from 'ol/source';
import { Feature} from "ol";
import { Point } from "ol/geom";
import { Draw, Snap } from 'ol/interaction';
import { fromLonLat } from "ol/proj";
import GeoJSON from 'ol/format/GeoJSON';
import { v4 as uuidv4 } from 'uuid';
import { Circle as CircleStyle, Fill, Stroke, Style } from 'ol/style';
export const pointSource = new VectorSource();
// Load and process point data from JSON
fetch('/js/points.json')
  .then(response => {
    if (!response.ok) {
      throw new Error("Failed to load points.json");
    }
    return response.json();
  })
  .then(pointDict => {
    addPointsToLayer(pointDict);
  })
  .catch(error => {
    console.error("Error loading JSON file:", error);
  });

// Function to add points to the map layer
function addPointsToLayer(pointDict) {
  Object.entries(pointDict).forEach(([imageId, labelDict]) => {
    // Loop over label colors/classes (e.g., "red", "green")
    Object.entries(labelDict).forEach(([label, coords]) => {
      if (!Array.isArray(coords)) {
        console.warn(`Skipping label "${label}" in image "${imageId}": coordinates are not an array.`);
        return;
      }

      coords.forEach(coord => {
        if (!Array.isArray(coord) || coord.length !== 2) {
          console.warn(`Invalid coordinate for label "${label}" in image "${imageId}":`, coord);
          return;
        }

        const [lat, lon] = coord;

        const pointFeature = new Feature({
          geometry: new Point(fromLonLat([lon, lat])),
          group: label,
          imageId: imageId,
          id: crypto.randomUUID()
        });

      // Optionally style the feature by group
      pointFeature.setStyle(getStyleForLabel(label));

      pointSource.addFeature(pointFeature);
    });
  });
})};
function getStyleForLabel(label) {
  const colorMap = {
    red: 'red',
    green: 'green',
    blue: 'blue'
    // add more as needed
  };

  return new Style({
    image: new CircleStyle({
      radius: 5,
      fill: new Fill({
        color: colorMap[label] || 'black'
      }),
      stroke: new Stroke({
        color: 'white',
        width: 1
      })
    })
  });
}
export const pointLayer = new VectorLayer({ title: "Area Layer", source: pointSource, style: new Style({
    fill: new Fill({
      color: 'rgba(255, 255, 255, 0.2)' 
    }),
    stroke: new Stroke({
      color: '#ffcc33', 
      width: 2
    }),
    image: new CircleStyle({
      radius: 7,
      fill: new Fill({
        color: '#ffcc33' 
      }),
      stroke: new Stroke({
        color: '#f39c12', 
        width: 1
      })
    })
  })});
const colorMap = {}; // assign a color per group

function getColorForGroup(group) {
  if (!colorMap[group]) {
    const hue = Math.floor(Math.random() * 360);
    colorMap[group] = `hsl(${hue}, 70%, 50%)`;
  }
  return colorMap[group];
}

pointLayer.setStyle(function (feature) {
  const group = feature.get('group');
  const color = getColorForGroup(group);

  return new Style({
    image: new CircleStyle({
      radius: 6,
      fill: new Fill({ color }),
      stroke: new Stroke({ color: '#333', width: 1 })
    })
  });
});