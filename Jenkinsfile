pipeline {
    agent any

    environment {
        IMAGE_NAME = "smart-attendance"
    }

    stages {

        stage('Clone GitHub Repo') {
            steps {
                git branch: 'main',
                url: 'https://github.com/SuhasRaj-21/DevOps-lab-01.git'
            }
        }

        stage('Install Dependencies') {
            steps {
                bat 'pip install -r requirements.txt'
            }
        }

        stage('Dependency Check') {
            steps {
                bat 'pip list'
            }
        }

        stage('SonarQube Analysis') {
            steps {
                echo 'Running SonarQube Analysis'
            }
        }

        stage('Docker Build') {
            steps {
                bat 'docker build -t smart-attendance .'
            }
        }

        stage('Run Docker Container') {
            steps {
                bat 'docker run -d -p 5000:5000 smart-attendance'
            }
        }
    }
}
