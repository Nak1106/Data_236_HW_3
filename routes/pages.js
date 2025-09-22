// routes/pages.js
const express = require('express');
const router = express.Router();

function requireAuth(req, res, next) {
  if (!req.session.user) {
    return res.redirect('/login');
  }
  next();
}

router.get('/', (req, res) => {
  res.render('index', { user: req.session.user, message: req.query.message });
});

router.get('/dashboard', requireAuth, (req, res) => {
  res.render('dashboard', { user: req.session.user });
});

module.exports = router;
