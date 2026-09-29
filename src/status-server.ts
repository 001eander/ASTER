import { createServer, type Server } from "node:http";
import { buildStatus, renderStatusHtml, renderStatusPanel } from "./status.js";

export type StatusServer = {
  url: string;
  close: () => Promise<void>;
};

export async function startStatusServer(runDir: string, port = 8787): Promise<StatusServer> {
  const server = createServer(async (req, res) => {
    try {
      const view = await buildStatus(runDir);
      if (req.url === "/api/status") {
        res.writeHead(200, { "content-type": "application/json; charset=utf-8" });
        res.end(JSON.stringify(view));
        return;
      }
      if (req.url === "/panel") {
        res.writeHead(200, { "content-type": "text/html; charset=utf-8" });
        res.end(renderStatusPanel(view));
        return;
      }
      res.writeHead(200, { "content-type": "text/html; charset=utf-8" });
      res.end(renderStatusHtml(view));
    } catch (err) {
      res.writeHead(500, { "content-type": "text/plain; charset=utf-8" });
      res.end(err instanceof Error ? err.message : String(err));
    }
  });

  await listen(server, port);
  const address = server.address();
  if (!address || typeof address === "string") throw new Error("status server has no port");
  return {
    url: `http://127.0.0.1:${address.port}`,
    close: () =>
      new Promise((resolve, reject) => {
        server.close((error) => (error ? reject(error) : resolve()));
        // The status page polls over a keep-alive connection; without this,
        // server.close() waits forever and the process never exits.
        server.closeAllConnections();
      }),
  };
}

function listen(server: Server, port: number): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(port, "127.0.0.1", () => resolve());
  });
}
