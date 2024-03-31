import express from "express";
import cors from "cors";
import open from 'open';
import fs from 'fs';

const app = express();
app.use(express.json());


app.use(cors())

app.get("/", (req, res) => {
    const command = "conda activate ml && python main.py"
    const script = `start cmd /k "${command}"`
    fs.writeFileSync("runScripts.bat",script)
    open('runScripts.bat',{wait:false})
});


app.listen(5000, () => {
    console.log("Connected to host 5000")
  

    
});


