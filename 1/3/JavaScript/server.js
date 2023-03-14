var http = require("http");

var counter = 1;

var onRequest = function(request, response) {
  response.writeHead(200, {"Content-Type": "text/plain"});
  response.write("Hello World!" + counter);
  
  function sleep(milliSeconds) {
    var startTime = new Date().getTime();
    while (new Date().getTime() < startTime + milliSeconds);
  }

  if (counter == 1) {
    sleep(10000);
  }
  
  counter++;
  response.end();
}

var server = http.createServer(onRequest);

server.listen(8888); 

// для работы с MySQL используйте пакет mysql2 и подготовленные запросы (prepared statements).
