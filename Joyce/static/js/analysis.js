var url = "https://us-gun-sale-and-violence-report-api.azurewebsites.net/api/v1.0/completedata"
var url_2 = "https://us-gun-sale-and-violence-report-api.azurewebsites.net/api/v1.0/statecoords"

d3.json(url_2).then(function(data){
    console.log(data)

    data.forEach(st =>{
        d3.select("#state_dropdown").append('option').attr('value',st.State).text(st.State)
    })
});

// d3.csv("../Resources/completed_data.csv").then(function(comp){

//     console.log(typeof comp.incident_date);
//     var year_2019_info = comp.incident_date.filter(endsWith("-19"));

//     console.log(year_info)

//     var grid = d3.select("#2019");
//         grid.html("");
//         Object.entries(year_2019_info[0]).forEach(i=> {
//             grid.append("p").text(`${i[0]}: ${i[1]}`)
//         });  
// })

function stateChanged(sta){
    d3.json(url).then(function(info){
        var kll=[]
        var inj=[]
        info.forEach(k =>{
            if (k.feature.State == sta){
                kll.push(k.feature.Killed)
                inj.push(k.feature.Injured)
            }
            else if (sta == "All"){
                kll.push(k.feature.Killed)
                inj.push(k.feature.Injured)
            }
        })
        var killed=0
        var injured=0
        var affected=0
        const nuevo = kll.map((i) => Number(i));
        console.log(nuevo);
        if (nuevo.length !== 0) {
        const reducer = (accumulator, curr) => accumulator + curr;
            killed=nuevo.reduce(reducer)}
            else{killed=0}

        const nue = inj.map((i) => Number(i));
        console.log(nue);
        if (nue.length !== 0) {
        const red = (accumulator, curr) => accumulator + curr;
            injured=nue.reduce(red)}
            else{injured=0}
        affected= killed+injured
        console.log(affected)

        var gauge1 = [{
            type: "indicator",
            mode: "gauge+number",
            value: killed,
            title: { text: "People Killed", font: { size: 20 } },
            gauge: {
                axis: { range: [null, 180], tickwidth: 1, tickcolor: "darkblue" },
                bar: { color: "red" },
                bgcolor: "white",
                borderwidth: 2,
                bordercolor: "gray",
                steps: [
                    { range: [0, 20], color: "#ffffff"},
                    { range: [20, 40], color: "#ffefde" },
                    { range: [40, 60], color: "#ffdebe" },
                    { range: [60, 80], color: "#ffd6ad" },
                    { range: [80, 100], color: "#ffc68d" },
                    { range: [100, 120], color: "#ffb66d" },
                    { range: [120, 140], color: "#ffae5d" },
                    { range: [140, 160], color: "#ff9e3a" },
                    { range: [160, 180], color: "#ff8e00"}
                ],
                threshold: {
                    line: { color: "red", width: 3 },
                    thickness: 0.75,
                    value: killed
                }
            }
        }];

        var gauge2 = [{
            type: "indicator",
            mode: "gauge+number",
            value: injured,
            title: { text: "People Injured", font: { size: 20 } },
            gauge: {
                axis: { range: [null, 810], tickwidth: 1, tickcolor: "darkblue" },
                bar: { color: "red" },
                bgcolor: "white",
                borderwidth: 2,
                bordercolor: "gray",
                steps: [
                    { range: [0, 90], color: "white" },
                    { range: [90, 180], color: "aliceblue" },
                    { range: [180, 270], color: "lightcyan" },
                    { range: [270, 360], color: "lightblue" },
                    { range: [360, 450], color: "cyan" },
                    { range: [450, 540], color: "darkturquoise" },
                    { range: [540, 630], color: "royalblue" },
                    { range: [630, 720], color: "mediumblue" },
                    { range: [720, 810], color: "navy" }
                ],
                threshold: {
                    line: { color: "red", width: 3 },
                    thickness: 0.75,
                    value: injured
                }
            }
        }];
    
        var layout = {
        title: sta,
        paper_bgcolor: "WhiteSmoke",
        font: { color: "darkblue", family: "Arial", size: 20,  fontWeight: "bold"}
        };
        
        Plotly.newPlot('gauge1', gauge1, layout);
        Plotly.newPlot('gauge2', gauge2, layout);

        var year_2019_all = info.filter(x=> {return x.feature.Year==2019});
        var year_2019_state= []
        info.forEach(i =>{
            if (i.feature.State == sta){
                year_2019_state= year_2019_all.filter(y=> {return y.feature.State==sta})
            }
            else if (sta == "All"){
            year_2019_state=year_2019_all
            }
        })

        console.log(year_2019_state)
        var incident_count= year_2019_state.length
        console.log(incident_count)


        function findOcc(arr, key){
            let arr2 = [];
              
            arr.forEach((x)=>{
                 
              // Checking if there is any object in arr2
              // which contains the key value
               if(arr2.some((val)=>{ return val[key] == x.feature.City })){
                   
                 // If yes! then increase the occurrence by 1
                 arr2.forEach((k)=>{
                   if(k[key] === x.feature.City){ 
                     k["occurrence"]++
                   }
                })
                   
               }else{
                 // If not! Then create a new object initialize 
                 // it with the present iteration key's value and 
                 // set the occurrence to 1
                 let a = {}
                 a[key] = x.feature.City
                 a["occurrence"] = 1
                 arr2.push(a);
               }
            })
              
            return arr2
        }

        // var id =d3.
        var org=d3.selectAll("#list-home-list").nodes();
        

        //2019
        var arr= year_2019_state
        var key="City"
        var incident_list=findOcc(arr, key)
        incident_list.sort((a, b) => b.occurrence - a.occurrence)
        var sorted=incident_list.slice(0,1)
        console.log(sorted)

        if (sorted != 0){
        var top_city=sorted[0].City
        console.log(top_city)
        var shoootings=sorted[0].occurrence

        year_2019_state.map((i) => Number(i.feature.Killed));
        year_2019_state.sort((a, b) => b.feature.Killed - a.feature.Killed)
        var deadly=year_2019_state
        worst_19=deadly.slice(0,1)[0].IncidentID

        var affectedd= (year_2019_state.map((i)=>Number(i.feature.Injured)).reduce(function(a, b){
            return a + b
        }, 0)) + (year_2019_state.map((i) => Number(i.feature.Killed)).reduce(function(a, b){
            return a + b;
        }, 0))
        }
        else {
            incident_count=0
            affected =0
            top_city = "N/A"
            shoootings=0
            worst_19="N/A"
        }

        var summary = d3.select("#list-home-list");
        summary.html("");
        summary.append("p").text(`Incident Count: ${incident_count}`)
        summary.append("p").text(`People Affected: ${affectedd}`),
        summary.append("p").text(`Most Dangerous City: ${top_city}`),
        summary.append("p").text(`Number of shootings in ${top_city}: ${shoootings}`),
        summary.append("p").text(`Deadliest Incident: ${worst_19}`)

        //2020

        var year_2020_all = info.filter(x=> {return x.feature.Year==2020});
        var year_2020_state= []
        info.forEach(i =>{
            if (i.feature.State == sta){
                year_2020_state= year_2020_all.filter(y=> {return y.feature.State==sta})
            }
            else if (sta == "All"){
            year_2020_state=year_2020_all
            }
        })

        console.log(year_2020_state)
        var incident_count= year_2020_state.length
        console.log(incident_count)

        var arr= year_2020_state
        var key="city_county"
        var incident_list=findOcc(arr, key)
        incident_list.sort((a, b) => b.occurrence - a.occurrence)
        var sorted=incident_list.slice(0,1)
        console.log(sorted)

        if (sorted != 0){
        var top_city=sorted[0].city_county
        console.log(top_city)
        var shoootings=sorted[0].occurrence

        year_2020_state.map((i) => Number(i.feature.Killed));
        year_2020_state.sort((a, b) => b.feature.Killed - a.feature.Killed)
        var deadly=year_2020_state
        worst_20=deadly.slice(0,1)[0].IncidentID

        var affectedd= (year_2020_state.map((i)=>Number(i.feature.Injured)).reduce(function(a, b){
            return a + b
        }, 0)) + (year_2020_state.map((i) => Number(i.feature.Killed)).reduce(function(a, b){
            return a + b;
        }, 0))
        }
        else{
            incident_count=0
            affected =0
            top_city = "N/A"
            shoootings=0
            worst_20="N/A"
        }

        var summary = d3.select(org[1]);
        summary.html("");
        summary.append("p").text(`Incident Count: ${incident_count}`)
        summary.append("p").text(`People Affected: ${affectedd}`),
        summary.append("p").text(`Most Dangerous City: ${top_city}`),
        summary.append("p").text(`Number of shootings in ${top_city}: ${shoootings}`),
        summary.append("p").text(`Deadliest Incident: ${worst_20}`)

        //2021

        var year_2021_all = info.filter(x=> {return x.feature.Year==2021});
        var year_2021_state= []
        info.forEach(i =>{
            if (i.feature.State == sta){
                year_2021_state= year_2021_all.filter(y=> {return y.feature.State==sta})
            }
            else if (sta == "All"){
            year_2021_state=year_2021_all
            }
        })

        console.log(year_2021_state)
        var incident_count= year_2021_state.length
        console.log(incident_count)

        var arr= year_2021_state
        var key="city_county"
        var incident_list=findOcc(arr, key)
        incident_list.sort((a, b) => b.occurrence - a.occurrence)
        var sorted=incident_list.slice(0,1)
        console.log(sorted)


        if(sorted != 0){
        var top_city=sorted[0].city_county
        console.log(top_city)
        var shoootings=sorted[0].occurrence

        year_2021_state.map((i) => Number(i.feature.killed));
        year_2021_state.sort((a, b) => b.feature.killed - a.feature.killed)
        var deadly=year_2021_state
        worst_21=deadly.slice(0,1)[0].IncidentID
        var affectedd= (year_2021_state.map((i)=>Number(i.feature.Injured)).reduce(function(a, b){
            return a + b
        }, 0)) + (year_2021_state.map((i) => Number(i.feature.Killed)).reduce(function(a, b){
            return a + b;
        }, 0))
        }
        else{
            incident_count=0
            affected =0
            top_city = "N/A"
            shoootings=0
            worst_21="N/A"
        }

        var summary = d3.select(org[2]);
        summary.html("");
        summary.append("p").text(`Incident Count: ${incident_count}`)
        summary.append("p").text(`People Affected: ${affectedd}`),
        summary.append("p").text(`Most Dangerous City: ${top_city}`),
        summary.append("p").text(`Number of shootings in ${top_city}: ${shoootings}`),
        summary.append("p").text(`Deadliest Incident: ${worst_21}`)
        

        //overall

        var year_state= []
        info.forEach(i =>{
            if (i.feature.State == sta){
                year_state= info.filter(y=> {return y.feature.State==sta})
            }
            else if (sta == "All"){
            year_state=info
            }
        })

        console.log(year_state)
        var incident_count= year_state.length
        console.log(incident_count)

        var arr= year_state
        var key="city_county"
        var incident_list=findOcc(arr, key)
        incident_list.sort((a, b) => b.occurrence - a.occurrence)
        var sorted=incident_list.slice(0,1)
        console.log(sorted)

        if(sorted != 0){
        var top_city=sorted[0].city_county
        console.log(top_city)
        var shoootings=sorted[0].occurrence

        year_state.map((i) => Number(i.feature.Killed));
        year_state.sort((a, b) => b.feature.Killed - a.feature.Killed)
        var deadly=year_state
        worst_all=deadly.slice(0,1)[0].IncidentID
        }
        else{
            incident_count=0
            affected =0
            top_city = "N/A"
            shoootings=0
            worst_all="N/A"
        }

        var summary = d3.select(org[3]);
        summary.html("");
        summary.append("p").text(`Incident Count: ${incident_count}`)
        summary.append("p").text(`People Affected: ${affected}`),
        summary.append("p").text(`Most Dangerous City: ${top_city}`),
        summary.append("p").text(`Number of shootings in ${top_city}: ${shoootings}`),
        summary.append("p").text(`Deadliest Incident: ${worst_all}`)
    })
}

stateChanged("All")