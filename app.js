// app.js
require('dotenv').config();

const express = require('express');
const session = require('express-session');
const bodyParser = require('body-parser');
const path = require('path');

const authRouter = require('./auth');
const pagesRouter = require('./routes/pages');

const app = express();

const PORT = process.env.PORT || 3000;

// view engine setup
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));

// static files if you add any
app.use(express.static(path.join(__dirname, 'public')));

// parse form data
app.use(bodyParser.urlencoded({ extended: true }));

// trust proxy for secure cookies when deployed
// app.set('trust proxy', 1);

// session setup
app.use(session({
  name: 'ads.sid',
  secret: process.env.SESSION_SECRET || 'dev_change_this',
  resave: false,
  saveUninitialized: false,
  cookie: {
    httpOnly: true,
    sameSite: 'lax',
    secure: process.env.NODE_ENV === 'production',
    maxAge: 1000 * 60 * 30
  }
}));

// simple request logger for debug
app.use((req, res, next) => {
  console.log('Method', req.method, 'Path', req.path);
  next();
});

// mount routers
app.use(authRouter);
app.use(pagesRouter);

// 404 handler
app.use((req, res) => {
  res.status(404).send('Not Found');
});

// start server
app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
});
