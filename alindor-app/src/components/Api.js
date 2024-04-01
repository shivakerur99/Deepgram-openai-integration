import axios from "axios";

const Api=axios.create({
    baseURL: "http://localhost:8000", //for localhost connection with FastAPI backend
    // baseURL: "https://alindor-ev3t.onrender.com"

    // baseURL: "https://alindor-hm.onrender.com",
})

export default Api