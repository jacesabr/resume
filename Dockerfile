FROM node:20-alpine
WORKDIR /app
COPY . .
RUN npm install && npm install -g serve
ENV PORT=3000
EXPOSE 3000
CMD sh -c "serve -s . -p $PORT"
