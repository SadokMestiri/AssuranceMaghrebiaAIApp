FROM node:20-alpine

WORKDIR /app

COPY frontend/package.json ./
COPY frontend/. .
RUN npm install

COPY . .
COPY data ./data
RUN npm run build

EXPOSE 3000

CMD ["npm", "run", "start"]
