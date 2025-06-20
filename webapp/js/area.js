import { map } from "./map";
import { Vector as VectorLayer } from 'ol/layer';
import { Vector as VectorSource } from 'ol/source';
import { Draw, Snap } from 'ol/interaction';
import GeoJSON from 'ol/format/GeoJSON';
import { v4 as uuidv4 } from 'uuid';
import { Circle as CircleStyle, Fill, Stroke, Style } from 'ol/style';

const drawSource = new VectorSource();
const finishedSource = new VectorSource();

export const drawAreaLayer = new VectorLayer({ title: "drawarea",source: drawSource, style: new Style({
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
export const AreaLayer = new VectorLayer({ title: "Area Layer", source: finishedSource, style: new Style({
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

const geojsonFormat = new GeoJSON();
const API_BASE = 'http://localhost:8000/api/polygons';

document.addEventListener('DOMContentLoaded', loadAreas);

async function loadAreas() {
  try {
    const res = await fetch(API_BASE);
    const data = await res.json();
    const features = geojsonFormat.readFeatures(data, {
      featureProjection: map.getView().getProjection(),
      dataProjection: 'EPSG:4326'
    });
    finishedSource.addFeatures(features);
    console.log("loaded areas")
    map.removeLayer(AreaLayer)
    map.addLayer(AreaLayer)
  } catch (err) {
    console.error('Fehler beim Laden der Polygone:', err);
  }
}

async function saveAreas() {
  const features = finishedSource.getFeatures();
  const geojson = geojsonFormat.writeFeaturesObject(features, {
    featureProjection: map.getView().getProjection(),
    dataProjection: 'EPSG:4326'
  });
  try {
    await fetch(API_BASE, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(geojson)
    });
  } catch (err) {
    console.error('Fehler beim Speichern der Polygone:', err);
  }
}

let draw, snap;
function addInteractions() {
  draw = new Draw({ source: drawSource, type: 'Polygon' });
  map.addInteraction(draw);
  draw.on('drawend', evt => {
    const feature = evt.feature;
    feature.setId(uuidv4());
    drawSource.removeFeature(feature);
    finishedSource.addFeature(feature);
    saveAreas();
    console.log("Saved area")
  });

  snap = new Snap({ source: drawSource });
  map.addInteraction(snap);
}
function removeInteractions() {
  map.removeInteraction(draw);
  map.removeInteraction(snap);
}

const editBtn = document.getElementById('edit1');
let editing = false;
editBtn.addEventListener('click', () => {
  editing ? removeInteractions() : addInteractions();
  editing = !editing;
});

/*
// Event-Listener für den Save-Button zum Speichern der gezeichneten Geometrien
save.addEventListener("click", () => {
  const formatGeoJSON = new GeoJSON();
  
  source.forEachFeature((f) => {
    if (!f.getId()) {
      f.setId(uuidv4()); 
    }

    const featureGeoJSON = formatGeoJSON.writeFeature(f, {
      dataProjection: "EPSG4326",
      featureProjection: "EPSG:3857",
      decimals: 6
    });

    const featureObj = JSON.parse(featureGeoJSON);

    const data = {
      "uuid": f.getId(),
      "geometry": featureObj.geometry
    };

    // POST-Anfrage an den Server zum Speichern der Flächen
    fetch("http://localhost:8083/create-areas", {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json; charset=utf-8'
      },
      body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(data => {
      console.log("Success: ", data);
    })
    .catch((error) => {
      console.error("Error:", error);
    });
  });
});

// Funktion zum Abrufen und Darstellen der Flächen aus der Datenbank
function get_areas() {
  const area_source = new VectorSource({
    loader: function(extent, resolution, projection) {
      fetch('http://localhost:8083/get_areas')
        .then(response => response.json())
        .then(data => {
          data.features.forEach(item => {
            const coordinates = item.geometry.coordinates;
            const feature = new Feature({
              geometry: new Polygon(coordinates), 
              uuid: item.properties.uuid,
            });
            feature.setId(item.properties.uuid); 
            area_source.addFeature(feature);
          });
        })
        .catch(error => {
          console.error("Error loading features:", error);
        });
    }
  });
  return area_source;
}

// Layer für Flächen 
export const areaLayer = new VectorLayer({
  title: "areas",
  source: get_areas(),
  style: new Style({
    fill: new Fill({
      color: 'rgba(4, 146, 16, 0.2)' 
    }),
    stroke: new Stroke({
      color: 'black', 
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
  })
});*/