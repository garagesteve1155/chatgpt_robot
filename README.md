Image Server Setup Instructions:

1. Create a LINUX DigitalOcean droplet or similar vps and open the terminal.
2. Use command "screen -S server"
3. Use command "mkdir public_images"
4. Use command "cd public_images"
5. Use command "sudo ufw allow 8080/tcp"
6. Use command "sudo ufw reload"
7. Use command "nohup python3 -m http.server 8080 &"
8. Hold down CTRL and press a, then release CTRL and press d
9. Use command "screen -S image"
10. Use command "wget https://raw.githubusercontent.com/garagesteve1155/chatgpt_robot/main/image_server.py"
11. Use command "python3 image_server.py"

This server is used to provide an image url to ChatGPT instead of the base64 image data like before. This saves A LOT of tokens (The server pays for itself very quick)
