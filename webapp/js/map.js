import { Map, View } from 'ol';
import TileLayer from 'ol/layer/Tile';
import { fromLonLat } from 'ol/proj';
import { OSM, XYZ } from 'ol/source';
import './area';
import { AreaLayer, drawAreaLayer } from './area';
import { initDownloadPopup } from './overlay';
import { pointLayer } from './points';

// 1. Create the WMS (imagery) layer as a TileLayer
export const wmsLayer = new TileLayer({
  source: new XYZ({
    url:
      'https://server.arcgisonline.com/ArcGIS/rest/services/' +
      'World_Imagery/MapServer/tile/{z}/{y}/{x}',
    maxZoom: 19,
  }),
  visible: false,             // start it visible
  title: 'World Imagery',    // optional, if you build a layer list later
});
export const osmLayer = new TileLayer({ source: new OSM() })
// 2. Build the map
export const map = new Map({
  target: 'map',
  layers: [
    osmLayer,
    wmsLayer,
    drawAreaLayer,
    AreaLayer,
    pointLayer,
  ],
  view: new View({
    center: fromLonLat([1.074901, 44.501916]),
    zoom: 17,
  }),
});













initDownloadPopup(map)
// 3. Wire up your “toggleSidebar” button
const btn = document.getElementById('toggleSidebar');
// initialize button text
btn.textContent = wmsLayer.getVisible() ? 'Hide WMS' : 'Show WMS';

btn.addEventListener('click', () => {
  const currentlyVisible = wmsLayer.getVisible();
  wmsLayer.setVisible(!currentlyVisible);
  // update button label:
  btn.textContent = currentlyVisible ? 'Show WMS' : 'Hide WMS';
});
// 3. Sidebar toggle button
const btnLayerList = document.getElementById('btnLayerList');
const sidebar = document.getElementById('layerListSidebar');
btnLayerList.addEventListener('click', () => {
  sidebar.classList.toggle('open');
  if (sidebar.classList.contains('open')) {
    updateLayerList();
  }
});

// 4. Populate & manage the layer list
function updateLayerList() {
  const list = document.getElementById('layerList');
  list.innerHTML = '';  // clear existing

  // filter out the two base layers
  const layers = map.getLayers().getArray()
    .filter(l => l !== osmLayer && l !== wmsLayer && l!= drawAreaLayer);

  layers.forEach((layer, i) => {
    const li = document.createElement('li');
    li.draggable = true;
    li.dataset.index = i;
    li.visible = true

    // checkbox
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = layer.getVisible();
    cb.addEventListener('change', () => {
      layer.setVisible(cb.checked);
    });

    // title
    const label = document.createElement('span');
    label.textContent = layer.get('title') || `Layer ${i+1}`;

    li.appendChild(cb);
    li.appendChild(label);
    list.appendChild(li);
    console.log('just appended:', li);
    console.log(list.innerHTML);

    // drag-and-drop handlers
    li.addEventListener('dragstart', e => {
      li.classList.add('dragging');
      e.dataTransfer.effectAllowed = 'move';
      e.dataTransfer.setData('text/plain', i);
    });
    li.addEventListener('dragover', e => {
      e.preventDefault();
      li.classList.add('drag-over');
    });
    li.addEventListener('dragleave', () => {
      li.classList.remove('drag-over');
    });
    li.addEventListener('drop', e => {
      e.preventDefault();
      li.classList.remove('drag-over');
      const fromIndex = parseInt(e.dataTransfer.getData('text/plain'), 10);
      const toIndex = parseInt(li.dataset.index, 10);

      // reorder in map
      const coll = map.getLayers();
      const layerToMove = coll.item(fromIndex);
      coll.removeAt(fromIndex);
      coll.insertAt(toIndex, layerToMove);

      updateLayerList();  // refresh indices and UI
    });
    li.addEventListener('dragend', () => {
      li.classList.remove('dragging');
    });
  });
}