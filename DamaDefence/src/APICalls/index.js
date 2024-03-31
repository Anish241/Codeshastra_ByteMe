import axios from 'axios';

const getService = async () => {
    try {
        const response = await axios.get('http://localhost:5000/');
        return response.data;
    } catch (error) {
        console.error(error);
    }

};


export {
    getService
}