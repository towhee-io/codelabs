import axios from 'axios';
import { decode } from 'js-base64';

const encryptedToken = 'Z2hwX3VhVlZqSXo2eUlPcWd1S0hDT2lwQk10dkZtYjhFVzFBeTAyWQ==';

const axiosInstance = axios.create({
  baseURL: 'https://api.github.com/repos',
  timeout: 10000,
});

axiosInstance.interceptors.request.use(
  function (config) {
    return config;
  },
  function (error) {
    // Do something with request error
    return Promise.reject(error);
  }
);


export const getGithubStatis = async () => {
  try {
    const res = await axiosInstance.get(`/milvus-io/milvus`, {
      headers: {
        Authorization: decode(encryptedToken)
      }
    });
    return res.data;
  } catch (error) {
    console.log(error);
  }
};

export default axiosInstance;
