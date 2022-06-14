import Head from 'next/head';
import '../../styles/fonts.css';
import '../../styles/globals.css';
import '../../styles/mixins.less';
import '../../styles/reset.css';
import '../../styles/variables.css';

function MyApp({ Component, pageProps }) {
  return (
    <>
      <Head>
        <script
          async
          src="https://www.googletagmanager.com/gtag/js?id=G-0JXY6PQLWQ"
        ></script>
        <script
          dangerouslySetInnerHTML={{
            __html: `window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'G-0JXY6PQLWQ');`,
          }}
        />
      </Head>
      <Component {...pageProps} />
    </>
  );
}

export default MyApp;
