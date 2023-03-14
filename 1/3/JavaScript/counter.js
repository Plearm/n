var http = require("http");

var counter = 1;

http.createServer(function(request, response) {
  response.writeHead(200, {"Content-Type": "text/plain"});
  response.write("Counter: " + counter);
  counter++;
  response.end();
}).listen(8888); 
