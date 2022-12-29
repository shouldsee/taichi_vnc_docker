__doc__ = '''
/GCAT/

This is an application aiming at simplifying 
particular types of GROMACS simulation of protein structures

## CHANGELOG and PLAN:

### v0.0.3
- [ ] adds api to evaluate a model against a pulling trajectory.
    - start a new flask server for pytorch
    - use pytorch for now
    - flowchart
      - load model to gpu
      - evaluate on test sets
      - calculate gradient for samples in traj
      - loss function with FAPE? (fape is great to avoid alignment)
    - dogefold.grad_with_xtc( model_weights, train_data, test_data, nstep)
      - load weights from pointer
      - parse xtc trajectories into paired frames of backbone coords
    - function to fold back 
- [ ] compile an appropriate pdb datasets for fitting the folding dynamics
- [ ] align the pull vec to tear the protein faster?
- [ ] adding consul.ext to git

### v0.0.2
- [x] adjust mdp simulation param to sample finer slicing of pulling
- [x] /p_md_simple_pull/ Adds endpoint to do steerred molecular dynamics
  - <https://manual.gromacs.org/current/reference-manual/special/pulling.html#the-transformation-pull-coordinate>
  - <https://manual.gromacs.org/documentation/current/user-guide/mdp-options.html#com-pulling>
  - CATSMILE-3008
  - CATSMILE-3012
- [] Adds interface to browse and search existing results
- [] simpler interface to investigate failed pypes
- [] extends templates. at the moment using non-inheritable strings

### v0.0.1
- [x] first impl of /p_md_simple
- [x] infrastructures to run and save simulations

### Unexpected
- [x] [FIXED]. NACOS connection is delayed until first request to flask.
flask does not admit a post_init hook. @app.before_first_request.

## Deployment Steps

- install docker,docker-compose
- change ports, hosts.
- run start.sh in the same directory as compose.yaml

## REF
- GROMACS CHN manual <https://jerkwin.github.io/9999/12/31/GROMACS%E4%B8%AD%E6%96%87%E6%89%8B%E5%86%8C/>
- GROMACS user comm <https://gromacs.bioexcel.eu/t/com-pulling/783>
- NACOS doc <https://nacos.io/zh-cn/docs/open-api.html>
- NACOS python sdk <https://github.com/nacos-group/nacos-sdk-python>
- CONSUL SWAGGER git: <https://github.com/Vakoto/hashicorp-consul-api>
  - SWAGGER <https://consul.docs.apiary.io/>
- CONSUL DOC <https://developer.hashicorp.com/consul/api-docs/health>

'''.strip()
#NacosClient.add_naming_instance

__version__ = 'v0.0.2'
# __version__ = 'v0.0.2'
import hashlib
import os,sys

from server_util import add_url_params
# import nacos

# import time
import threading



import flask
from datetime import datetime
#import flask
def str2hex(x,encoding='utf-8'):
    return (hashlib.md5(x.encode(encoding)).hexdigest())

from apiflask import APIFlask, Schema, abort, APIBlueprint
from apiflask.fields import Integer, String
from apiflask.validators import Length, OneOf

from flask import ( 
    url_for,render_template_string,
    render_template,
    redirect,
    request,
    )
import flask_wtf
from flask_wtf import Form
from wtforms import StringField
import wtforms as wtf


import mimetypes
# [DOC] https://github.com/wtforms/flask-wtf/blob/main/src/flask_wtf/form.py
# [DOC] https://github.com/wtforms/wtforms/blob/3.0.x/src/wtforms/form.py
# [DOC] List of WTFfields https://wtforms.readthedocs.io/en/3.0.x/fields/


from pype.controller import PypeExecResult, Controller


# app0 = APIFlask(__name__, static_folder='/shared_data/', static_url_path = '/shared_data/')

app0 = APIFlask(__name__)


app = bp = APIBlueprint('root', __name__)



app0.config['PREFIX_GCAT'] = PREFIX_GCAT = prefix = '/gcat/'
APP_NAME = PREFIX_GCAT.strip('/')
# app0.config['NGL_APP_URL'] = NGL_APP_URL = 'http://localhost:9001/mdsrv/webapp/' 
app0.config['NGL_APP_URL'] = NGL_APP_URL = os.environ['NGL_APP_URL']
# 'http://192.168.50.132:9001/mdsrv/webapp/' 
app0.config['APP_VERSION']=__version__
app0.config['APP_STARTED']=datetime.now().isoformat()




def ahref_from_url(url,extra=""):
    return f'<a href="{url}" {extra}>{url}</a>'

from consul.ext import ConsulExt, ConsulExtService
# from consul.base import Check

def init_register_service(app=app0):
    '''
    URL=192.168.50.132:9006
    SERVICE_ID=gcat_gromacs
    curl -XPUT http://${URL}/v1/agent/service/deregister/${SERVICE_ID}
    # '''

    x = ConsulExtService(
        os.environ['CONSUL_URL'],
        os.environ['SERVICE_NAME'],None,
        address= os.environ['SERVICE_IP'],
        port = int(os.environ['SERVICE_PORT'])
        )
    app.config.registry_connected =  x.register_service_strict()
    x.start_register_thread()
    x.start_heartbeat_thread()                
    app.regclient=x

init_register_service()


# @app0.route('/shared_data/', defaults={'req_path': ''})
@app0.get('/shared_data/<path:req_path>')
def dir_listing(req_path):

    BASE_DIR = '/shared_data/'
    req_path = req_path.rstrip('/')
    # Joining the base and the requested path
    abs_path = os.path.join(BASE_DIR, req_path)

    if not os.path.exists(abs_path):
        # Return 404 if path doesn't exist
        return flask.abort(404)
    elif os.path.isfile(abs_path):
        # Check if path is a file and serve
        # return flask.send_file(abs_path)
        mimetype,encoding = mimetypes.guess_type(abs_path)
        mimetype = 'text/plain' if mimetype is None else mimetype
        # assert 0,mimetype
        return flask.send_file(abs_path,as_attachment=False,mimetype=mimetype)
        # return flask.send_file(abs_path,as_attachment=False,mimetype='text/plain')
    else:   
        # return [os.path.isfile(abs_path),abs_path]
        # print()
        # Show directory contents
        files = os.listdir(abs_path)
        return render_template_string(
            '''
    <ul>
        {% for file in files %}
        <li>
            <a href="{{ request.path + ['/',''][(request.path.endswith('/'))] + file }}">
                {{ request.path + ['/',''][(request.path.endswith('/'))] + file }}
            </a>
        </li>
        {% endfor %}
    </ul>        
            ''', files=files)



from collections import namedtuple
import glob
import simplejson as json
from simplejson import JSONDecoder, JSONDecodeError
import re
NOT_WHITESPACE = re.compile(r'[^\s]')
def json_decode_stacked(document, pos=0, decoder=JSONDecoder()):
    '''
    Source: https://stackoverflow.com/a/50384432/8083313
    '''
    while True:
        match = NOT_WHITESPACE.search(document, pos)        
        if not match:
            return
        pos = match.start()
        
        try:
            obj, pos = decoder.raw_decode(document, pos)
        except JSONDecodeError:
            # do something sensible if there's some error
            raise
        yield obj

@app.get('/')
def index_page():
    # [str(p) for p in temp_app.url_map.iter_rules()]
    return render_template_string('''
    <a href="{{PREFIX_GCAT}}"><h3>GCAT {{app.config.APP_VERSION}}</h3></a>

    <pre>
    APP_STARTED: {{app.config.APP_STARTED}}
    debug:{{debug}}
    registry_connected:{{app.config.registry_connected}}
    registry_client:{{app.regclient.url}}
    last_beat_good:{{app.regclient.last_beat_good}}
    last_beat_good_ts:{{app.regclient.last_beat_good_ts}}
    last_beat:{{app.regclient.last_beat}}
    {% if not app.regclient.last_beat_good %}
    last_beat_detail: {{app.regclient.check_in_json}}
    {% endif %}

    </pre>
 

    <h5>DOC</h5>
    <pre>{{__doc__}}
    </pre>


    <h4>GETable url</h4>
    {% for x in app.url_map.iter_rules() %}

        <!--
            {{ x.__dict__.keys()     |safe }}
            <p>{{x}}</p>
            <p>{{x.__repr__()}}</p>
            <pre>{{x.__doc__}}</pre>                        
        -->
    

        <table border="1">
        <tbody>
            {% if debug or ("GET" in x.methods and not x.arguments) %}
            <tr>
                <td>

                    {{ahref_from_url(x.rule) | safe}}
                    {% if debug %}
                        <table border="1">
                        <tbody>
                        {% for k in "rule endpoint host methods arguments ".split() %}
                                <tr>
                                    <td>
                                    {{ k }}
                                    </td>
                                    <td>
                                    {{ x.__dict__[k]     |safe }}
                                    </td>
                                </tr>
                        {% endfor %}
                        </tbody></table>
                    {% endif %}

                </td>
            </tr>
            {% endif %}
        </tbody></table>

    {% endfor %}

    ''',app=app0, debug=int(request.args.get('debug',0)), ahref_from_url=ahref_from_url,__doc__ = __doc__)


PypeJsonData =namedtuple('PypeJsonData','path job_name shex msgs')
class PypeJsonData(PypeJsonData):
    def is_ended(self):
        return self.msgs[-1]['name'] == '_PYPE_END'

    @classmethod
    def from_file_path(cls,xx):
        path = xx
        xx = xx[len(SDATA_DIR)+1:]
        sp = xx.split('/',2)
        with open(path,'r') as f:
            it = json_decode_stacked(f.read())
            parsed = list(it)
        xxx = cls(path=path,job_name=sp[0],shex= sp[1], msgs = parsed,)        
        return xxx
if 1:
    '''
    PP_JSONS is the in-memory index of on-disk tasks
    PP_JSONS should be modified to reflect changes of 
    '''
    SDATA_DIR = '/shared_data'
    x = glob.glob(SDATA_DIR+'/*/*/PYPE.json')
    ret = []
    for xx in x:
        # xxx =  PypeJsonData.from_file_path(xx)
        ret.append(PypeJsonData.from_file_path(xx))
        # xxx)
    PP_JSONS = ret

ENV_RUNDIR = '/shared_data'


    

def p_md_generic(request, action_name, form_cls):
    IS_DEBUG_VIEW = int(request.args.get('IS_DEBUG_VIEW',False))
    rundir = request.args.get('rundir', ENV_RUNDIR)
    error = ""
    action_base, action_name_new = request.endpoint.rsplit('.',1)
    if action_name is None:
        action_name = action_name_new
    form = form_cls(request.form)
    if request.method=='POST':
        ret = Actions.job_submit(action_name, form.data, rundir)
        ret_msg = render_template_string('''

    <table border="1">
    <thead></thead>
    <tbody>
    {% for k,v in ret.items() %}
            <tr>
                <td>
                {{ k }}
                </td>
                <td>
                {% if k not in "data output_data".split() %}
                    {{ v |safe }}
                {% else %}
                    <table border="1">
                        <tbody>            
                        {% for kk,v in ret.get(k,{}).items() %}
                                <tr>
                                    <td>
                                    {{ kk }}
                                    </td>
                                    <td>
                                    {{ v |safe }}
                                    </td>
                                </tr>
                        {% endfor %}
                    </tbody></table>
                {%endif%}
                </td>
            </tr>
    {% endfor %}
    <tr>
        <td>
        output_data
        </td>
        <td>

        </td>
    </tr>
    </tbody></table>
        ''',ret=ret,action_base=action_base,action_name=action_name)
    else:
        ret_msg = '<p>empty since NOT a POST request</p>'

    resp = render_template_string('''

<html>
<head><title>{{form.__class__.__name__}}</title>
</head>
    <h3>
    {{form.__class__.__name__}}
    </h3>
    {{IS_DEBUG_VIEW}}
    {{action_name}}
    {{request.view_args}}
    <pre>{{form.__doc__}}
    </pre>
    <table border="1">
    <thead></thead>
    <tbody>
    </tbody></table>


{# <form method="POST" action="{{action_base}}job_submit/{{form.job_name}}"> #}
<form method="POST" action="">

    {{ form.csrf_token }}


    <table border="1">
    <thead></thead>
    <tbody>
    {% for k,v in form._fields.items() %}
            <tr>
                <td>
            {{v.label}}
                </td>
                <td>
            {{v(size=20)}}
                </td>
            </tr>

    {% endfor %}

    </tbody></table>
    <input type="submit" value="Submit">

</form>

<br>
<h3>Returned Message</h3>

{{ ret_msg | safe }}
<br>


<h3>Loaded Results </h3>
    <table border="1">
    <thead></thead>
    <tbody>

    {% for v in PP_JSONS %}
      <tr>
      {% for f in "job_name shex path".split() %}
                <td>
                {{v[f]}}
                </td>
      {% endfor %}

      <td>
        {{ v.is_ended() }}
      </td>

      {% if IS_DEBUG_VIEW %}
      <td>
        {{ v.msgs[0:2] }}
      </td>
      <td>
        {{ v.msgs[-1:] }}
      </td>
      {% endif %}

    </tr>
    {% endfor %}

    </tbody>
    </table>

</html>
        ''',form=form,PP_JSONS=PP_JSONS, IS_DEBUG_VIEW=IS_DEBUG_VIEW,ret_msg=ret_msg,
        action_base = action_base,action_name=action_name)
    return resp


class SimpleMdMessage(Schema):
    '''
    `struct` is a path to a 
    '''
    struct = String(required=True, validate=Length(0, 150))
    md_steps = Integer()


class SimpleMdResult(Schema):
    result_pointer = String()

import threading 
tdict = {}
@app.get('/job_status/<string:action_name>/<string:shex>')
# @app.input(SimpleMdMessage)
def job_status(action_name, shex,json=json):
    tkey = action_name + '/' + shex
    data = tdict[tkey]
    t = data['thread']
    is_alive = t.is_alive()
    datajson = json.dumps(data,indent=2,default=lambda x:x.__repr__())
    return flask.render_template_string('''
    <html>
        <body>
        <p>

        Job submitted!

        When the calculation is done, you can access it at 

        {{datajson}}
        </p>
        <pre>
        </pre>
        <table border="1">
        <thead></thead>
        <tbody>

            <tr>
                <td>
                    is_alive
                </td>
                <td>
                    {{is_alive}}
                </td>
            </tr>


            <tr>
                <td>
                    ts_start
                </td>
                <td>
                    {{data["ts_start"]}}
                </td>
            </tr>

            <tr>
                <td>
                    ret
                </td>
                <td>
                    {{data["ret"]}}
                </td>
            </tr>

        </tbody>
        </table>
        </body>        
        </html>''', **locals())    

    #  data):

# @app.post('/job_submit/<string:action_name>')
# @app.input(SimpleMdMessage,location='json_or_form')
# def job_submit(action_name,data):
#     """
#     submit a job to a targeted
#     """
#     # data = request.data
#     data0 = data.copy()
#     is_ret_input = int(request.args.get('is_ret_input',0))
#     if is_ret_input:
#         return data0
#     # print(data0)

#     return Actions.job_submit(action_name, data)




class SimpleMdMessageForm(Form):
    '''
    Run a molecular simple dynamics simulation with gromacs
    Need to supply `struct`

    `TBC:DONE:impl a status checking page to show whether the long running process. probably with a term-like interface. or with PYPE.json` 
    `TBC:adding persistence mechanism to build index for existing tasks, probably with PYPE.json`


    `struct` is a path to a 
    '''
    struct = StringField()
    # required=True, validate=Length(0, 150))
    md_steps = wtf.IntegerField()
    # job_name = "md_simple"
    # md_steps = Integer()

class p_md_simple_pull_form(Form):
    '''
    Run a molecular simple dynamics simulation with gromacs
    Need to supply `struct`

    `TBC:DONE:impl a status checking page to show whether the long running process. probably with a term-like interface. or with PYPE.json` 
    `TBC:adding persistence mechanism to build index for existing tasks, probably with PYPE.json`

    md_pull_coord1_k: spring const in kJ mol^-1 nm^-2 negative for pulling,
      positive for squeezing

    `struct` is a path to a 
    '''
    struct               = StringField()
    md_steps             = wtf.IntegerField()
    md_log_interval_step = wtf.IntegerField(default=5000)
    md_dt_fs             = wtf.DecimalField(default=0.002,places=10)
    md_pull_coord1_k     = wtf.DecimalField(default=-500)
    box_radius           = wtf.DecimalField(default=3.0,places=10)


@app.route('/p_md_simple',methods='GET POST'.split())
def p_md_simple():
    return p_md_generic(request,None, SimpleMdMessageForm)


@app.route('/p_md_simple_pull',methods='GET POST'.split())
def p_md_simple_pull():
    return p_md_generic(request,None, p_md_simple_pull_form)


import inspect
def check_dict_for_func(data, func, ignore_kw=None, target='raise', ):
    '''
    :param:
      target: control the returned value 
        raise: raise error if detected
        noerror: return True if no error else False
        full: return a dictionary with key POSITIONAL_MISSING DATA_EXTRA
          to describe whether the data meets function signature
    Returns:
    '''
    xpar = inspect.signature(func).parameters
    x = {}
    if ignore_kw is None:
        ignore_kw = []
    elif isinstance( ignore_kw ,str):
        ignore_kw = ignore_kw.split()

    assert target in 'raise noerror full'.split(),target    

    x['POSITIONAL'] = []
    x['KEYWORD'] = []
    x['VAR_KEYWORD'] = []
    x['VAR_POSITIONAL'] = []
    for k,v in xpar.items():
        if k in ignore_kw:
            continue
        kname = v.kind.name
        if kname == 'POSITIONAL_OR_KEYWORD':
            kname = 'KEYWORD POSITIONAL'.split()[int(v.default is inspect._empty)]
        x[kname].append(v)
    
    ### detect unmet arguments
    x['POSITIONAL_MISSING'] = []
    for v in x['POSITIONAL']:
        if v.name not in data:
            x['POSITIONAL_MISSING'].append(v)
    noerr = True
    noerr &= len(x['POSITIONAL_MISSING'])==0
    if (target=='raise') and not noerr:            
        raise Exception( f'Dictionary missing arguments {x["POSITIONAL_MISSING"]} for calling {func!r}' )
    x['POSITIONAL_OR_KEYWORD'] = [xx.name for xx in x['POSITIONAL'] + x['KEYWORD']]

    ### detect excessive arguments
    x['DATA_EXTRA']= []
    for k in data:
        if (k not in x['POSITIONAL_OR_KEYWORD']) :
            x['DATA_EXTRA'].append(k)

    noerr &= (len(x['DATA_EXTRA'])==0 or len(x['VAR_KEYWORD'])!=0 )
    if (target=='raise') and not noerr:            
        raise Exception( f'Dictionary contains excessive arguments {x["DATA_EXTRA"]} for calling {func!r}' )        
    x['NO_ERROR']= noerr

    if target=='full':
        return x
    elif target in 'noerror raise'.split():
        return noerr
    else:
        raise NotImplementedError(target)

class Actions(object):
    from know_simple_md import know_simple_md
    @classmethod
    def p_md_simple(cls, data, rundir,check_only):
        return cls.p_md_generic(cls.know_simple_md, data, rundir, check_only)


    from know_simple_md_pull import know_simple_md_pull
    @classmethod
    def p_md_simple_pull(cls, data, rundir,check_only):
        return cls.p_md_generic(cls.know_simple_md_pull, data,rundir, check_only)


    @classmethod
    def job_submit(cls, action_name, data, rundir):
        '''
        If the job is already in cache, then just return the cache
        '''
        rundir = rundir + '/' + action_name
        act = getattr(cls,action_name)
        data['shex'] = shex = str2hex(json.dumps(sorted(data.items())))
        tkey = f'''{action_name}/{data['shex']}'''
        pype_json_file = f'''{rundir}/{data['shex']}/PYPE.json'''

        ret_data = dict(
            message = "Job submitted!", 
            data=data, 
            output_data = {},
            tkey=tkey,
            url_status = ahref_from_url(url_for(f'{PREFIX_GCAT}.job_status',action_name=action_name,shex=shex)),
            pype_json_file=ahref_from_url(pype_json_file)
            )
            
        check_only = False
        if os.path.exists(pype_json_file):
            pdata = PypeJsonData.from_file_path(pype_json_file) 
            if pdata.is_ended():
                ret_data['message'] = 'Job cache found and returned!'
                check_only = True

        def wrapper(data,tkey=tkey,act=act,rundir=rundir,check_only=check_only):
            try:


                tdict[tkey]['ret'] = ret_data['output_data'] = act( data, rundir, check_only)
                # xxx = PypeJsonData(path=path,job_name=sp[0],shex= sp[1], msgs = parsed,)        
                # ret.append(xxx) 
                PP_JSONS.append(PypeJsonData.from_file_path(pype_json_file))               
            except Exception as e:
                tdict[tkey]['error'] = e
                raise e

        if tdict.get(tkey,None) is not None and tdict[tkey]['thread'].is_alive():
            ret_data['message'] = 'Job thread found running. please wait and check job_status'
        else:
            t = threading.Thread(target=wrapper,args=(data,))
            tdict[tkey] = dict(input_data=data,thread=t)
            t.start()            
            if check_only:
                t.join()
        return ret_data

    @staticmethod
    def _normalize_struct_url(struct):
        if struct.startswith('file://'):
            struct = struct[len('file://'):]
            assert os.path.exists(struct),struct
            # assert 
            raise NotImplementedError('The pype recipe is not compatible with file:// url ')
        elif struct.startswith('rcsb://'):
            struct = struct[len('rcsb://'):]
        else:
            assert NotImplementedError(struct)
        return struct


    @classmethod
    def p_md_generic(cls, action_obj, data, rundir,check_only,):
        """

        The sole idea is that the parameter decides which function to call

        Option 1: clone and proxy the request 
        - Pros
            - proxy is useful tech
            - And yes I am making another request to a internal 
        - Cons
            - information may be lost during proxying
            - the same request will be parsed multiple times
        - work: 5/10
        Option 2: Refactor to get callable from the registry 
        - Pros
            - fast
        - Cons
            - not sure 


        <http://localhost:9001/mdsrv/webapp/?struc=file://static-prot/1PGB/5123456/md_0_1.gro&traj=file://static-prot/1PGB/5123456/md_0_1.xtc&data=/mdsrv/&mdsrv=/mdsrv/>
        <http://localhost:9001/mdsrv/webapp/?struc=/mdsrv/file/static-prot/1PGB/5123456/md_0_1.gro&traj=file://static-prot/1PGB/5123456/md_0_1.xtc&data=/mdsrv/&mdsrv=/mdsrv/>

        """    
        

        param    = data.copy()
        data['outdir'] = rundir+'/'+data['shex']

        ### Param validation
        data['PDB_ID']  = cls._normalize_struct_url(data['struct'])
        shex            = data['shex']
        md_steps        = data['md_steps']
        data['GMX']     = 'gmx'

                                

            ### check all
            
        if not check_only:
            check_dict_for_func(data,action_obj,'ctl'.split())
            pype2 = Controller.from_func(action_obj, **data)
                # # GMX    = pype1.built["GMX"],
                # GMX      = data['GMX'],
                # PDB_ID   = data['PDB_ID'],
                # md_steps = data['md_steps'],
                # )

            # pype2 = Controller.from_func(action_obj, 
            #     # GMX    = pype1.built["GMX"],
            #     GMX      = data['GMX'],
            #     PDB_ID   = data['PDB_ID'],
            #     md_steps = data['md_steps'],
            #     )

            # outdir = rundir+'/'+struct + '/' + str(md_steps)
            pype2.run(rundir=data['outdir'],extra_nodes=[
                    PypeExecResult(name='GCAT_PARAM',suc=1,extra_dict=param),
                    # PypeExecResult(name='WEB_PARAM', suc=1,extra_dict=param)
            ])

        outdir = data['outdir']
        reldir = outdir.lstrip('/')
        fodir = 'file://' + reldir
        url_ngl = add_url_params( 
            app0.config['NGL_APP_URL'],
            {
            "struc":f'{fodir}/md_0_1.gro',
            'traj':f'{fodir}/md_0_1.xtc',
            'data':'/mdsrv/',
            'file':'/mdsrv/'})

        data['url_ngl']  = url_ngl
        data['url_ngl_href']  = f'<a href="{url_ngl}">{url_ngl}</a>'
        data['rundir']   = rundir
        return data






app = app0
# Swagger UI
app.config['SWAGGER_UI_CSS'] = 'https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/4.11.1/swagger-ui.min.css'
app.config['SWAGGER_UI_BUNDLE_JS'] = 'https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/4.11.1/swagger-ui-bundle.min.js'
app.config['SWAGGER_UI_STANDALONE_PRESET_JS'] = 'https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/4.11.1/swagger-ui-standalone-preset.min.js'
# Redoc
app.config['REDOC_STANDALONE_JS'] = 'https://cdn.redoc.ly/redoc/latest/bundles/redoc.standalone.js'
# Elements
app.config['ELEMENTS_JS'] = 'https://cdn.jsdelivr.net/npm/@stoplight/elements-dev-portal@1.7.4/web-components.min.js'
app.config['ELEMENTS_CSS'] = 'https://cdn.jsdelivr.net/npm/@stoplight/elements-dev-portal@1.7.4/styles.min.css'
# RapiDoc
app.config['RAPIDOC_JS'] = 'https://cdn.jsdelivr.net/npm/rapidoc@9.3.2/dist/rapidoc-min.min.js'
# RapiPDF
app.config['RAPIPDF_JS'] = 'https://cdn.jsdelivr.net/npm/rapipdf@2.2.1/src/rapipdf.min.js'

app.register_blueprint(bp, url_prefix=prefix,name=prefix)





