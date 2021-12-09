# UAUQE

<div id="top"></div>

<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="images/logo_smart _puj.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">UAQUE: Sistema de Recomendación basado en el Perfil Grupal</h3>

  <p align="center">
    <br />
    <a href="https://www.youtube.com/watch?v=C0TYQPvs5qk">Ver video de explicación y Demo</a>
    ·
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Tabla de contenido</summary>
  <ol>
    <li>
      <a href="#Acerca-del-proyecto">Acerca del proyecto</a>
      <ul>
        <li><a href="#built-with">Tecnologías usadas</a></li>
      </ul>
    </li>
    <li>
      <a href="#Servicios-Smart-UJ-relacionados">Servicios involucrados en el contexto Smart UJ</a>
    </li>
    <li>
      <a href="#getting-started">Información de instalación</a>
      <ul>
        <li><a href="#Instalación Prerequisitos">Instalación y Prerequisitos</a></li>
      </ul>
    </li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## Acerca del proyecto

UAQUE es un proyecto destinado a desarrollar un servicio inteligente para la Pontificia Universidad Javeriana sede Bogotá que permita realizar recomendaciones de material bibliográfico de la Biblioteca Alfonso Borrero Cabal, S.J. de acuerdo con el perfil grupal de las personas de la comunidad académica en el marco de un campus inteligente.

Para logar este objetivo se realizo el desarrollo de un modelo de analítica y un conjunto de dashboards que permiten generar las recomendaciones y obtener información de las recomendaciones.

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- RELATED SERVICES -->
## Servicios Smart-UJ relacionados

#### SMART-UJ Api Gateway
Enfocada en exponer los servicios desarrollados por UAQUE y expuestos a continuación.

#### SMART-UJ UI Uaque
Interfaces de usuario. Al tratarse de un sistema enfocado en ser usado por dos tipos de usuarios existe una interfaz diferente para cada uno. 
Para la comunidad académica está la aplicación móvil, enfocada en: 
<br>
Configurar las notificaciones de las recomendaciones
<br>
Mostrar las notificaciones de las recomendaciones
<br>
Usar el formulario para definir los gustos de los usuarios nuevos
<br>
Configurar el canal para recibir las notificaciones
<br>
Para los funcionarios de la biblioteca existe el dashboard, enfocado en:
<br>
Visualizar los datos de las recomendaciones y el proceso de clustering
<br>
Configurar las URLs de los archivos que alimentan el modelo de analítica para producir las recomendaciones

#### SMART-UJ Ubicacion por red
Servicio usado para activar las notificaciones que sean configuradas para mostrarse al momento de estar cerca de la biblioteca.

#### SMART-UJ Uso biblioteca
Servicio enfocado en consumir y procesar los datos anonimizados de las transacciones de material bibliográfico registradas por la biblioteca, así como la información del material bibliográfico registrado.

#### SMART-UJ Recomendacion de tematicas por grupo UAQUE
Servicio encargado de generar las recomendaciones por temática Dewey a partir de las transacciones de material bibliográfico y la información de este material bibliográfico y los grupos producidos por el servicio de perfil grupal.

#### SMART-UJ Perfil de usuario
Servicio enfocado en elaborar el perfil de usuario, usado en el proceso de clustering y consumido más adelante por el servicio de perfil grupal.

#### SMART-UJ Perfil grupal
Servicio que se alimenta del perfil de usuario para llevar a cabo un proceso de clustering que permite agrupar usuarios por temática y más adelante generar las recomendaciones.

### Herramientas de desarrollo

A continuación, se listan los frameworks y librerías usadas para el desarrollo de UAQUE

* [Python](https://www.python.org/)
* [Scikit-learn - librería(Python)](https://scikit-learn.org/stable/)
* [Jupyter](https://jupyter.org/)
* [React](https://reactjs.org/)
* [Postman](https://www.postman.com/)


<p align="right">(<a href="#top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Información de instalación

Las instrucciones para replicar el proceso de instalación y los prerequisitos necesarios se encuentran en el siguiente documento: 
* [Manual completo de instalación](https://github.com/Johan-Ortegon-1/uaque/blob/main/documents/Manual_de_Instalacion_Completo.pdf)

<p align="right">(<a href="#top">back to top</a>)</p>
