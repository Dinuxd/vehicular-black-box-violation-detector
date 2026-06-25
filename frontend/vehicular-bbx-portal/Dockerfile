FROM caddy:2.9-alpine

# Set Caddy to use Temp to write files.
ENV XDG_DATA_HOME=/tmp/caddy

# Set working directory
WORKDIR /app

# Copy the built React app into the container
COPY dist /app
COPY Caddyfile /etc/caddy/Caddyfile

EXPOSE 8080

CMD ["caddy", "run", "--config", "/etc/caddy/Caddyfile"]