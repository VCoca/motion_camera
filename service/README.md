# Pokretanje kao servis

Da bi sistem radio po uključenju ploče, bez prijave i bez terminala.

```bash
sudo cp service/motion-camera.service /etc/systemd/system/
sudo cp service/motion-web.service    /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now motion-camera motion-web
```

Provera i praćenje:

```bash
systemctl status motion-camera
journalctl -u motion-camera -f
```

Zaustavljanje za vreme razvoja, da servis ne drži GPIO liniju i kameru:

```bash
sudo systemctl stop motion-camera
```

`WorkingDirectory` je zadat jer program putanje izvodi iz položaja
`config.py`, a servis se inače pokreće iz korena datotečnog sistema.
Ako se projekat premesti, izmeniti putanje u obe jedinice.
