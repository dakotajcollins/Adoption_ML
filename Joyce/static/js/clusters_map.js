// Creating the map object
var myClustersMap = L.map("Cmap", {
  center: [40.27883153043306, -101.96116805914728],
  zoom: 5
});

  // Adding the tile layer
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
  }).addTo(myClustersMap);

  var url = "https://us-gun-sale-and-violence-report-api.azurewebsites.net/api/v1.0/completedata";

// Get the data with d3.
d3.json(url).then(function(response) {

    console.log(response)

    // Create a new marker cluster group.
    var markers = L.markerClusterGroup();
  
    // Loop through the data.
    for (var i = 0; i < response.length; i++) {
  
      // Set the data location property to a variable.
      var location = response[i].feature;
  
      // Check for the location property.
      if (location) {
  
        // Add a new marker to the cluster group, and bind a popup.
        markers.addLayer(L.marker([location.Latitude, location.Longitude])
          .bindPopup(location.Date+" "+location.Year +"<hr>City:"+ location.City+"<br>Injured:"+ location.Injured+"<br> Killed:"+ location.Killed));
      }
  
    }
  
    // Add our marker cluster layer to the map.
    myClustersMap.addLayer(markers);
  
  });