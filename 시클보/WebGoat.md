# WebGoat 실습
## 목차
1. A1 Injection  
    1-1 [(A1) Injection(intro)](#sql-injectionintro)  
    1-2 [(A2) SQL Injection(advanced)](#sql-injectionadvanced)  
## (A1) Injection
### SQL Injection(intro)
<b>[2 개념]</b>  
SQL(Structured Query Language)이란?  
: 관계형 databases를 관리하고 database 안에서 data에서 다양한 작업을 수행하기 위해 사용되는 표준화된(ANSI in 1986, ISO in 1987) 프로그래밍 언어이다.  

database : data의 모임 <br>   
data : 행들, 열들, 테이블들로 구성됨 그리고 관련 정보를 쉽게 찾을 수 있도록 indexing 됨  
<br>  

employees 테이블  <br>  
<img src = "../image/SQL[2] 문제.png" width ="100%">

[2의 문제]  <br>
아래 Employees Table에서 Bob Franco의 부서를 이끌어 내라  <br>

[2의 답]  
<img src = "../image/SQL[2]답.png" width ="100%">

<b>[3 개념]</b>  
DML(Data Manipulation Language)란?  
: 테이블 안에 있는 데이터를 다룬다.  

- SELECT : 하나의 database로부터 데이터를 도출한다.
- INSERT : 하나의 table에 데이터를 삽입한다.
- UPDATE : 하나의 table 내부에 존재하는 데이터를 update(수정) 한다.  
- DELETE : 한 database table로부터 모든 기록들을 삭제한다.  

[3의 문제]   
Tobi Barnett의 부서를 'Sales'로 부서를 바꿔봅시다. <br>
[3의 답]
<img src = "../image/SQL[3]답.png" width ="100%">

<b>[4 개념]</b>  
DDL(Data Definition Language)란?  
: 데이터를 정의하는 언어  
: 어떤 데이터 베이스, 스키마, 테이블이 필요한지 등 데이터의 구조를 설계  

- CREATE : database, 테이블 등을 생성
- ALTER : 존재하는 database, 테이블의 구조를 수정한다.
- DROP : database, 테이블을 삭제

[4의 문제]  
employees 테이블에 "phone" column(varchar(20))을 추가하여 scheme를 수정해봅시다.  <br>
[4의 답]
<img src = "../image/SQL[4]답.png" width ="100%">

<b>[5 개념]</b>  
DCL(Data Control Language)란?  
: 데이터 사용 권한을 주는 언어




### SQL Injection(advanced)