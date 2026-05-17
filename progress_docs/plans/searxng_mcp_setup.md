# SearXNG MCP Setup Plan

## Problem Summary

SearXNG was running but the JSON API returned **403 Forbidden** when accessed. This was caused by multiple issues:

1. `network_mode: host` in docker-compose.yaml doesn't work on Windows
2. JSON format was not enabled in SearXNG settings
3. The `formats` setting was incorrectly placed under `server:` instead of `search:`

## Solution Applied

### 1. Modified `searxng/settings.yml`

Added JSON format support under the correct section:

```yaml
search:
  safe_search: 0
  default_lang: ""
  formats:
    - html
    - json
```

### 2. Modified `docker-compose.yaml`

Changed Caddy from `network_mode: host` to bridge networking with explicit ports:

```yaml
caddy:
  container_name: caddy
  image: docker.io/library/caddy:2-alpine
  restart: unless-stopped
  ports:
    - "127.0.0.1:80:80"
    - "127.0.0.1:443:443"
  networks:
    - searxng
```

### 3. Modified `Caddyfile`

Changed reverse proxy from localhost to container name:

```
reverse_proxy searxng:8080
```

### 4. Added MCP Configuration

Added SearXNG MCP server to `mcp_settings.json`:

```json
{
  "searxng": {
    "command": "npx",
    "args": ["-y", "mcp-searxng"],
    "env": {
      "SEARXNG_URL": "http://127.0.0.1:8080"
    }
  }
}
```

## Final Configuration Files

### `searxng/settings.yml`

```yaml
use_default_settings: true

server:
  secret_key: "90f977076fe78b28dab8a748ed0cd75aa785fd66d78ddace77bba5e343be92d4"
  limiter: false
  image_proxy: true
  trusted_proxies:
    - 0.0.0.0/0

search:
  safe_search: 0
  default_lang: ""
  formats:
    - html
    - json

redis:
  url: redis://redis:6379/0

outgoing:
  request_timeout: 10.0
  max_request_timeout: 15.0
  pool_connections: 100
  pool_maxsize: 20
  enable_http2: true
```

### `docker-compose.yaml`

```yaml
services:
  caddy:
    container_name: caddy
    image: docker.io/library/caddy:2-alpine
    restart: unless-stopped
    ports:
      - "127.0.0.1:80:80"
      - "127.0.0.1:443:443"
    volumes:
      - ./Caddyfile:/etc/caddy/Caddyfile:ro
      - caddy-data:/data:rw
      - caddy-config:/config:rw
    environment:
      - SEARXNG_HOSTNAME=${SEARXNG_HOSTNAME:-localhost}
      - SEARXNG_TLS=${LETSENCRYPT_EMAIL:-internal}
    networks:
      - searxng
    logging:
      driver: "json-file"
      options:
        max-size: "1m"
        max-file: "1"

  redis:
    container_name: redis
    image: docker.io/valkey/valkey:8-alpine
    command: valkey-server --save 30 1 --loglevel warning
    restart: unless-stopped
    networks:
      - searxng
    volumes:
      - valkey-data2:/data
    logging:
      driver: "json-file"
      options:
        max-size: "1m"
        max-file: "1"

  searxng:
    container_name: searxng
    image: docker.io/searxng/searxng:latest
    restart: unless-stopped
    networks:
      - searxng
    ports:
      - "127.0.0.1:8080:8080"
    volumes:
      - ./searxng:/etc/searxng:rw
      - searxng-data:/var/cache/searxng:rw
    environment:
      - SEARXNG_BASE_URL=https://${SEARXNG_HOSTNAME:-localhost}/
    logging:
      driver: "json-file"
      options:
        max-size: "1m"
        max-file: "1"

networks:
  searxng:

volumes:
  caddy-data:
  caddy-config:
  valkey-data2:
  searxng-data:
```

## Access URLs

- **HTTP (Direct)**: http://127.0.0.1:8080
- **HTTPS (via Caddy)**: https://localhost

Both endpoints now support the JSON API format:
- `curl "http://127.0.0.1:8080/search?q=test&format=json"`
- `curl -k "https://localhost/search?q=test&format=json"`

## Next Steps

1. **Restart VS Code** to load the new MCP server configuration
2. The SearXNG MCP server will be available for web searches
3. Test by asking Roo to perform a web search

## Troubleshooting

If the JSON API returns 403:
1. Check that `formats: [html, json]` is under `search:` section, not `server:`
2. Restart the searxng container: `docker compose restart searxng`
3. Check logs: `docker compose logs searxng --tail=50`

If Caddy can't connect to SearXNG:
1. Ensure both containers are on the same network: `docker network inspect searxng-docker_searxng`
2. Check Caddy logs: `docker compose logs caddy --tail=50`

---

## Status (from setup-searxng-mcp.md — 2026-04-26)

Partially complete. SearXNG MCP server configured in `.kilocode/mcp.json`. Docker dependency being evaluated. Alternative: hosted SearXNG instance or direct API usage.

| # | Task | Status |
|---|------|--------|
| 1 | Evaluate Docker vs hosted SearXNG | ⏳ |
| 2 | Configure MCP server connection | 🔄 Partial |
| 3 | Test search queries from agent | ⏳ |
| 4 | Document in COMMAND_CHEATSHEET.md | ⏳ |

Note: Alternative search tools already available (Tavily, Exa).
