// Use the D3 library to read in samples.json.


//Read the url input the data into State drop down select tag
url="https://us-gun-sale-and-violence-report-api.azurewebsites.net/api/v1.0/gunsalebystate"
var url_1 = "https://us-gun-sale-and-violence-report-api.azurewebsites.net/api/v1.0/completedata"



d3.json(url).then(function createPlotly(data) {
    console.log(data);

    var state= []
    data.forEach(function(i){
      state.push(i.State)
    })
    console.log(state);

    // var statevalue=[]
    // data.forEach(function(v){
    // statevalue.push(v.State,v.feature['2017'],v.feature['2018'],v.feature['2019'],v.feature['2020'],v.feature['2021'])
    // })
    // console.log(statevalue);

   
    d3.select("#selDataset_state").selectAll("option").data(state).enter().append("option")
    .html(function(d) {
    return `<option>${d}</option>`;
    });


    // State Dropdown menu
    var dropdownMenu = d3.select("#selDataset_state");
    var dropdownValue = dropdownMenu.property("value");
    var index = state.indexOf(dropdownValue);

    // Start Year Dropdown menu
    var InputMenu_SY = d3.select("#selDataset_startyear");
    var InputValue_SY = InputMenu_SY.property("value");

    // End Year Dropdown menu
    var InputMenu_EY = d3.select("#selDataset_endyear");
    var InputValue_EY = InputMenu_EY.property("value");
    


    // Create a bar graph using index
    var Data2017 = data[index].feature['2017']
    var Data2018 = data[index].feature['2018']
    var Data2019 = data[index].feature['2019']
    var Data2020 = data[index].feature['2020']
    var Data2021 = data[index].feature['2021']
    var defaultData = [Data2017,Data2018,Data2019,Data2020,Data2021]
    console.log(defaultData)
    

    var label=['2017','2018','2019','2020','2021']

    var bardata = [
      // {
      //   x: defaultData,
      //   y: label,
      //   type: "bar",
      //   orientation: "h",
      // }

      {
        x: label,
        y: defaultData,
        type: "bar",

      }
    ];
  
    var barLayout = {
      title: `Gun Sale in ${dropdownValue}`,
      xaxis: { title: "Year" }
    };
  
    Plotly.newPlot("bar", bardata, barLayout);

    
    //Line chart
    var gdata
        graphing = function(state,start_date,end_date)
        {
        
          start_date_format = new Date(start_date);
          end_date_format = new Date(end_date);

        d3.json(url_1).then(function(response) {
        gdata = response
        // console.log(response)
        var killed = []
        var injured = []
        var dates = []
        statedata = gdata.filter(d => {
        var monthday = d.feature.Date;
        var year = d.feature.Year;
        var date = new Date(monthday +', '+year)
        return (d.feature.State==state && date>=start_date_format && date<=end_date_format)
        })
        console.log(statedata)
    
        injuredsort = statedata.sort((a,b)=>{
          var monthday = a.feature.Date;
          var year = a.feature.Year;
          var adate = new Date(monthday +', '+year)
          var monthday = b.feature.Date;
          var year = b.feature.Year;
          var bdate = new Date(monthday +', '+year)
          return adate-bdate
        })
        console.log(injuredsort)

        injuredsort.slice(0,20)
        killed = injuredsort.slice(0,20).map(d=>d.feature.Killed)
        injured = injuredsort.slice(0,20).map(d=>d.feature.Injured)
        dates = injuredsort.slice(0,20).map(d=>d.feature.Date+", "+d.feature.Year)
    
        console.log(injuredsort.slice(0,20))
        console.log(killed)
        console.log(injured)
        console.log(dates)

      
        const line = {
          labels: dates,
          datasets: [
              {
                  name: "Amount Killed",
                  values: killed,
                  chartType: 'bar'
              },
              {
                  name: "Amount Injured",
                  values: injured,
                  chartType: 'bar'
              },
              {
                name: "Amount Killed",
                values: killed,
                chartType: 'line'
              },
              {
                name: "Amount Injured",
                values: injured,
                chartType: 'line'
              },
          ]
        }
        
        const linechart = new frappe.Chart("#LineBarchart", {  
                                                 
          title: `Crime from Gun Violence ${dropdownValue}`,
          data: line,
          type: 'axis-mixed', 
          height: 350,
          colors: ['#7cd6fd', '#743ee2']
        })
      });
      };
    graphing(dropdownValue,InputValue_SY,InputValue_EY)


  //map
    var myMap = L.map("DashBoard_map", {
      center: [39.8283, -98.5795],
      zoom: 4
    });
    
    // Adding the tile layer and initial marker set
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    }).addTo(myMap);
    
    var markersLayer = new L.LayerGroup();
    markersLayer.clearLayers()
    mapping(dropdownValue,InputValue_SY,InputValue_EY)
    updateMap()

    function updateMap() {
      state_dropdown = d3.select("#selDataset_state");
      state = state_dropdown.property("value");
      console.log(state);
      
      start_dropdown = d3.select("#selDataset_startyear");
      start_date = new Date(start_dropdown.property("value"));
      console.log(start_date);
    
      end_dropdown = d3.select("#selDataset_endyear");
      end_date = new Date(end_dropdown.property("value"));
      console.log(end_date);
      
      markersLayer.clearLayers();
      updateview(state);
      mapping(state,start_date,end_date)
    
    
    }

    function updateview(state){

      var state_cord = Object.values(state_cord_data);
      var state_name = Object.keys(state_cord_data);
      var center_cord = [39.8283, -98.5795];
      var zoom_num = 4;
      for (var i = 0; i < state_name.length; i++) {
        if(state_name[i]==state)
        {
          var center_latitude = state_cord[i].Latitude;
          var center_longitude = state_cord[i].Longitude;
          var center_cord = [center_latitude,center_longitude]
          var zoom_num = 6      
        }
      };
      // myMap.setZoom(zoom_num);
      myMap.flyTo(center_cord,zoom_num);
      console.log(center_cord);
      
      };
      //create a function that maps the markers
        function mapping(state,start_date,end_date)
        {
        // Get the data with d3.
        d3.json(url_1).then(function(response) {

          // Create a new marker cluster group.
          var markers = L.markerClusterGroup();

          // Loop through the data.
          for (var i = 0; i < response.length; i++) {

            var monthday = response[i].feature.Date;
            var year = response[i].feature.Year
            var date = new Date(monthday +', '+year)
            var state1 = response[i].feature.State;
            var incident = response[i].IncidentID;
            var injured = response[i].feature.Injured;
            var killed= response[i].feature.Killed;
            var incident_url = 'https://www.gunviolencearchive.org/incident/' + incident


            if (response[i].feature.State==state && date>=start_date && date<=end_date)
            {
              // Set the data location property to a variable.
              var latitude = response[i].feature.Latitude;
              var longitude = response[i].feature.Longitude;
              

              // Check for the location property.
              if (latitude) {
              // Add a new marker to the cluster group, and bind a popup.
              markers.addLayer(L.marker([latitude, longitude]).bindPopup('Date: ' + date.toDateString() 
              + '</br> State: ' + state1
              + '</br> # Injured: ' + injured
              + '</br> # Killed: ' + killed
              + '</br> <a href="' + incident_url+ '" target="popup">More Info</a>'
              ));
              }
            }
          
            else if (state=='All' && date>=start_date && date<=end_date)
            {
                // Set the data location property to a variable.
              var latitude = response[i].feature.Latitude;
              var longitude = response[i].feature.Longitude;
              

              // Check for the location property.
              if (latitude) {
              // Add a new marker to the cluster group, and bind a popup.
              markers.addLayer(L.marker([latitude, longitude]).bindPopup('Date: ' + date.toDateString() 
              + '</br> State: ' + state1
              + '</br> # Injured: ' + injured
              + '</br> # Killed: ' + killed
              + '</br> <a href="' + incident_url+ '" target="popup">More Info</a>'
              ));
              }
            }

          }

        // Add our marker cluster layer to the map.
        markersLayer = markers
        
        myMap.addLayer(markersLayer);

        });

        }
 






    
    // When different test ID is selected, call an function optionChanged
   d3.select("#selDataset_state").on("change", optionChanged);
  
    function optionChanged() {
     console.log("Different State was selected.");
     var dropdownMenu = d3.select("#selDataset_state");
     var dropdownValue = dropdownMenu.property("value");
     console.log(`Currently state ${dropdownValue} is shown on the page`);

     // Update graph
     myMap.off();
     myMap.remove();
     createPlotly(data);
    
   }


   d3.select("#selDataset_startyear").on("change", optionChanged);
  
   function optionChanged() {
    console.log("Start Date was changed.");
    var InputMenu_SY = d3.select("#selDataset_startyear");
    var InputValue_SY = new Date(InputMenu_SY.property("value"));
    console.log(`Currently start date ${InputValue_SY} is shown on the page`);


    
     // Update graph
     myMap.off();
     myMap.remove();   
     createPlotly(data);

   }

   d3.select("#selDataset_endyear").on("change", optionChanged);
  
   function optionChanged() {
    console.log("End Date was changed.");
    var InputMenu_EY = d3.select("#selDataset_endyear");
    var InputValue_EY = new Date(InputMenu_EY.property("value"));
    console.log(`Currently end state ${InputValue_EY} is shown on the page`);



 
     // Update graph
     myMap.off();
     myMap.remove();
     createPlotly(data);

   }



  })







    

    

    

    
    


 
   