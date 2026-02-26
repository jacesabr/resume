const http = require("http");
const fs = require("fs");
const path = require("path");

const PORT = process.env.PORT || 3000;

// Startup diagnostics
console.log("=== STARTUP ===");
console.log("PORT:", PORT);
console.log("__dirname:", __dirname);
console.log("files:", fs.readdirSync(__dirname).filter(f => f.endsWith(".html")).length, "html files");
console.log("index.html exists:", fs.existsSync(path.join(__dirname, "index.html")));

const MIME = {
  ".html": "text/html", ".css": "text/css", ".js": "application/javascript",
  ".json": "application/json", ".png": "image/png", ".jpg": "image/jpeg",
  ".svg": "image/svg+xml", ".ico": "image/x-icon", ".woff": "font/woff",
  ".woff2": "font/woff2", ".txt": "text/plain",
};

const server = http.createServer((req, res) => {
  console.log(req.method, req.url);
  let url = decodeURIComponent(req.url.split("?")[0]);
  if (url === "/") url = "/index.html";
  if (!path.extname(url)) url += ".html";

  const file = path.resolve(path.join(__dirname, url));
  if (!file.startsWith(__dirname)) {
    res.writeHead(403);
    return res.end("Forbidden");
  }

  fs.readFile(file, (err, data) => {
    if (err) {
      fs.readFile(path.join(__dirname, "index.html"), (e, d) => {
        res.writeHead(e ? 404 : 200, { "Content-Type": "text/html" });
        res.end(e ? "Not Found" : d);
      });
      return;
    }
    res.writeHead(200, { "Content-Type": MIME[path.extname(file)] || "application/octet-stream" });
    res.end(data);
  });
});

server.listen(PORT, "0.0.0.0", () => {
  console.log("=== SERVER READY on 0.0.0.0:" + PORT + " ===");
});

server.on("error", (err) => {
  console.error("SERVER ERROR:", err);
});
