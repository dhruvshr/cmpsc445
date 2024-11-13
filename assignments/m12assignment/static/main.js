function getResult() {
    var url = "http://localhost:8000";
    var endpoint = "/result";
    var http = new XMLHttpRequest();

    // prepare GET request
    http.open("GET", url+endpoint, true);

    http.onreadystatechange = function() {
        var DONE = 4;
        var OK = 200;
        if (http.readyState == DONE && http.status == OK && http.responseText) {

            // parse the json response
            var replyObj = JSON.parse(http.responseText);

            // display the results
            document.getElementById("result").innerHTML =
                // "Model:     " + replyObj.model + "<br>" +
                "Accuracy:  " + replyObj.accuracy + "<br>" +
                "Precision: " + replyObj.precision + "<br>" +
                "Recall:    " + replyObj.recall;
        }

    };
    // send request
    http.send();
}