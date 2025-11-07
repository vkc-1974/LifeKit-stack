* Create Livekit configuration (for local usage):
```shell
  `$ docker run -it --rm -v$(pwd):/output livekit/generate --local` 
```
  or use existing one
```yaml
port: 7880
rtc:
    udp_port: 7882
    tcp_port: 7881
    port_range_start: 50000
    port_range_end:   50999
keys:
    APIAr4ziPRxD7RQ: NG4BDigFkZpjXZrJ7oPfHd9p0WdxPLuffJcAKUHJjKfC
logging:
    json: false
    level: info
redis:
  address: redis:6379
```

* set up `uv` package manager:
```shell
pip install uv
```

* Set up environment variables (host context to be used by `lk` CLI utility):
```shell
export LIVEKIT_URL=ws://localhost:7880
export LIVEKIT_API_SECRET=NG4BDigFkZpjXZrJ7oPfHd9p0WdxPLuffJcAKUHJjKfC
export LIVEKIT_API_KEY=APIAr4ziPRxD7RQ
```

* Set up python dependencoes/requirements:
```shell
sudo dnf install mariadb-connector-c-devel
pip install --upgrade pip
pip install \
    "livekit-agents[all]" \
    "livekit-plugins-openai" \
    "livekit-plugins-deepgram" \
    mcp \
    mariadb \
    httpx \
    fastapi \
    uvicorn
```

* Prepare `docker-compse.yaml` file (`mariadb + Livekit + redis + livekit/sip`):
```yaml
services:
  # MariaDB
  mariadb:
    image: mariadb:11.4
    environment:
      MYSQL_ROOT_PASSWORD: rootpass
      MYSQL_DATABASE: livekit_db
      MYSQL_USER: livekit_user
      MYSQL_PASSWORD: livekit_pass
    ports:
      - "3316:3306"
    volumes:
      - mariadb_data:/var/lib/mysql
    restart: unless-stopped

  # Radis
  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
    restart: unless-stopped

  # Livekit SFU Server
  livekit:
    image: livekit/livekit-server:latest
    ports:
      - "7880:7880"
      - "7881:7881"
      - "7882:7882/udp"
      - "51000-51999:50000-50999/udp"
    volumes:
      - ./livekit.yaml:/etc/livekit.yaml
    command: --config /etc/livekit.yaml --dev
    depends_on:
      - redis
    restart: unless-stopped

  # Livekit SIP Server
  sip:
    image: livekit/sip:latest
    ports:
      - "5060:5060/tcp"
      - "5060:5060/udp"
      - "55000-55999:50000-50999/udp"
    environment:
      SIP_CONFIG_BODY: |
        log_level: debug
        api_key: APIAr4ziPRxD7RQ
        api_secret: NG4BDigFkZpjXZrJ7oPfHd9p0WdxPLuffJcAKUHJjKfC
        ws_url: ws://livekit:7880
        redis:
          address: redis:6379
        sip_port: 5060
        rtp_port: 50000-50999
        use_external_ip: true
    depends_on:
      - livekit
      - redis
    restart: unless-stopped

volumes:
  mariadb_data:
  redis_data:
```

* Start services (containers)
```shell
docker compose up
```

* Prepare database table and populate it:
```shell
docker exec -it $(docker ps -qf "name=mariadb") mariadb -u livekit_user -plivekit_pass livekit_db
```
```sql
CREATE TABLE users (
  user_id INT PRIMARY KEY,
  balance_value DECIMAL(10,2) NOT NULL
);
INSERT INTO users (user_id, balance_value) VALUES (123, 1000.50), (456, 2500.00);
SELECT * FROM users;
EXIT;
```

* Download LLM
```shell
docker exec -it $(docker ps -qf "name=ollama") ollama pull llama3.2:3b-instruct-q4_K_M
```

* Create dispatch rule
```shell
cat > dispatch-rule.json << EOF
{
  "dispatch_rule": {
    "rule": {
      "dispatchRuleIndividual": {
        "roomPrefix": "test-"
      }
    },
    "roomConfig": {
      "agents": [
        {
          "agentName": "voice-agent"
        }
      ]
    }
  }
}
EOF
lk sip dispatch create dispatch-rule.json
```
* Set up SIP client (`https://www.zoiper.com/en/voip-softphone/download/current`)
  SIP account (you must to use main host network interface IP address, not localhost):
      * Username: testtrunk
      * Password: testpass
      * Domain: 192.168.31.41:5060


