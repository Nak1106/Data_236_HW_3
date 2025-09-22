// auth.js
const express = require('express');
const router = express.Router();
const bcrypt = require('bcryptjs');

const users = [
  { id: 1, username: 'admin', password: bcrypt.hashSync('password', 8) }
];

router.get('/login', (req, res) => {
  res.render('login', { error: req.query.error });
});

router.post('/login', (req, res) => {
  const { username, password } = req.body;
  const user = users.find(u => u.username === username);
  if (user && bcrypt.compareSync(password, user.password)) {
    req.session.user = { id: user.id, username: user.username };
    return res.redirect('/dashboard');
  }
  res.redirect('/login?error=1');
});

router.get('/logout', (req, res) => {
  req.session.destroy(err => {
    if (err) return res.redirect('/dashboard');
    res.clearCookie('ads.sid');
    res.redirect('/');
  });
});

module.exports = router;
