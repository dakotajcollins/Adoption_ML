var myHeatMap = L.map("Hmap", {
  center: [40.27883153043306, -101.96116805914728],
  zoom: 5
});

// Adding the tile layer
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
}).addTo(myHeatMap);

var url = "https://us-gun-sale-and-violence-report-api.azurewebsites.net/api/v1.0/completedata";

d3.json(url).then(function(heat) {

  console.log(heat);

  var heatArray=[]

  for (var i = 0; i < heat.length; i++) {
    var location = heat[i].feature;

    if (location) {
      heatArray.push([location.Latitude, location.Longitude]);
    }
  }


  var heat = L.heatLayer(heatArray, {
        radius: 35,
        blur: 15, 
        maxZoom: 10,
        max: 4.0,

        gradient: {
            0.0:'green',
            0.3: 'blue',
            0.7: 'yellow',
            1.0: 'red'
        }
    
  }).addTo(myHeatMap);

});